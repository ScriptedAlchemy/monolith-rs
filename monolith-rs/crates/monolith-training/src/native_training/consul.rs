//! Python `monolith.native_training.consul` parity.
//!
//! The original Python implementation talks to a ByteDance Consul sidecar via
//! `/v1/lookup/name?name=...`. For Rust parity tests we keep the same request
//! paths and JSON formats, but implement a small pluggable HTTP transport so
//! tests can mock network behavior.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ConsulError(pub String);

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

pub trait HttpTransport: Send + Sync + 'static {
    fn request(
        &self,
        method: &str,
        host: &str,
        port: u16,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> std::io::Result<HttpResponse>;
}

/// Default best-effort HTTP/1.1 client over TCP (and UNIX sockets on Unix).
///
/// This is intentionally tiny and only supports what the Python parity code uses.
#[derive(Debug, Default)]
pub struct DefaultHttpTransport;

impl DefaultHttpTransport {
    fn request_tcp(
        &self,
        method: &str,
        host: &str,
        port: u16,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> std::io::Result<HttpResponse> {
        let addr = format!("{host}:{port}");
        let mut stream = std::net::TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;

        let body_len = body.map(|b| b.len()).unwrap_or(0);
        let mut req = Vec::new();
        write!(
            &mut req,
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Length: {body_len}\r\n\r\n"
        )?;
        if let Some(b) = body {
            req.extend_from_slice(b);
        }
        stream.write_all(&req)?;

        let mut resp = Vec::new();
        stream.read_to_end(&mut resp)?;
        parse_http_response(&resp)
    }

    #[cfg(unix)]
    fn request_unix(
        &self,
        method: &str,
        sock_path: &str,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> std::io::Result<HttpResponse> {
        use std::os::unix::net::UnixStream;
        let mut stream = UnixStream::connect(sock_path)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;

        let body_len = body.map(|b| b.len()).unwrap_or(0);
        let mut req = Vec::new();
        // "Host" header is required by many HTTP servers even over UDS.
        write!(
            &mut req,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {body_len}\r\n\r\n"
        )?;
        if let Some(b) = body {
            req.extend_from_slice(b);
        }
        stream.write_all(&req)?;

        let mut resp = Vec::new();
        stream.read_to_end(&mut resp)?;
        parse_http_response(&resp)
    }
}

impl HttpTransport for DefaultHttpTransport {
    fn request(
        &self,
        method: &str,
        host: &str,
        port: u16,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> std::io::Result<HttpResponse> {
        if host.starts_with('/') {
            #[cfg(unix)]
            {
                return self.request_unix(method, host, path, body, timeout);
            }
            #[cfg(not(unix))]
            {
                let _ = (method, host, port, path, body, timeout);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "unix sockets not supported on this platform",
                ));
            }
        }
        self.request_tcp(method, host, port, path, body, timeout)
    }
}

fn parse_http_response(buf: &[u8]) -> std::io::Result<HttpResponse> {
    // Minimal HTTP response parsing: status line + headers + body.
    let text = String::from_utf8_lossy(buf);
    let (head, body) = match text.split_once("\r\n\r\n") {
        Some(v) => v,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid HTTP response",
            ))
        }
    };
    let mut lines = head.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty response"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad status line"))?
        .parse::<u16>()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad status code"))?;

    Ok(HttpResponse {
        status,
        body: body.as_bytes().to_vec(),
    })
}

fn now_secs_f64() -> f64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    dur.as_secs_f64()
}

#[derive(Debug, Clone)]
struct CacheEntry {
    cachetime: f64,
    ret: Vec<serde_json::Value>,
}

/// A small Consul HTTP client matching the Python API surface area.
#[derive(Clone)]
pub struct Client {
    consul_sock: String,
    consul_host: String,
    consul_port: u16,
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    http: Arc<dyn HttpTransport>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("consul_sock", &self.consul_sock)
            .field("consul_host", &self.consul_host)
            .field("consul_port", &self.consul_port)
            .finish_non_exhaustive()
    }
}

impl Client {
    pub fn new() -> Self {
        Self::with_http(Arc::new(DefaultHttpTransport))
    }

    pub fn with_http(http: Arc<dyn HttpTransport>) -> Self {
        let consul_sock = "/opt/tmp/sock/consul.sock".to_string();
        let mut consul_host = std::env::var("CONSUL_HTTP_HOST")
            .ok()
            .or_else(|| std::env::var("TCE_HOST_IP").ok())
            .unwrap_or_default();
        if consul_host.is_empty() {
            if std::path::Path::new(&consul_sock).is_file() {
                consul_host = consul_sock.clone();
            } else {
                consul_host = "127.0.0.1".to_string();
            }
        }
        let consul_port = std::env::var("CONSUL_HTTP_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(2280);

        Self {
            consul_sock,
            consul_host,
            consul_port,
            cache: Arc::new(Mutex::new(HashMap::new())),
            http,
        }
    }

    pub fn lookup(
        &self,
        name: &str,
        timeout_secs: u64,
        cachetime_secs: u64,
    ) -> std::io::Result<Vec<serde_json::Value>> {
        let now = now_secs_f64();
        if cachetime_secs > 0 {
            if let Some(cached) = self
                .cache
                .lock()
                .expect("consul client cache mutex should not be poisoned")
                .get(name)
                .cloned()
            {
                if now - cached.cachetime <= cachetime_secs as f64 {
                    return Ok(cached.ret);
                }
                // When cache exists but expired, keep requested timeout.
                let ret = self._lookup(name, Duration::from_secs(timeout_secs))?;
                self.cache
                    .lock()
                    .expect("consul client cache mutex should not be poisoned")
                    .insert(
                    name.to_string(),
                    CacheEntry {
                        ret: ret.clone(),
                        cachetime: now,
                    },
                );
                return Ok(ret);
            }

            // When cache is missing, Python increases timeout to 30.
            let timeout = if timeout_secs == 0 { 30 } else { timeout_secs };
            let ret = self._lookup(name, Duration::from_secs(timeout))?;
            self.cache
                .lock()
                .expect("consul client cache mutex should not be poisoned")
                .insert(
                name.to_string(),
                CacheEntry {
                    ret: ret.clone(),
                    cachetime: now,
                },
            );
            return Ok(ret);
        }

        let ret = self._lookup(name, Duration::from_secs(timeout_secs))?;
        self.cache
            .lock()
            .expect("consul client cache mutex should not be poisoned")
            .insert(
            name.to_string(),
            CacheEntry {
                ret: ret.clone(),
                cachetime: now,
            },
        );
        Ok(ret)
    }

    fn _lookup(&self, name: &str, timeout: Duration) -> std::io::Result<Vec<serde_json::Value>> {
        let path = format!("/v1/lookup/name?name={name}&addr-family=dual-stack");
        let resp = self.http.request(
            "GET",
            &self.consul_host,
            self.consul_port,
            &path,
            None,
            timeout,
        )?;
        if resp.status != 200 {
            // Python logs and returns [].
            tracing::error!(
                status = resp.status,
                body = %String::from_utf8_lossy(&resp.body),
                "consul: request failed"
            );
            return Ok(Vec::new());
        }
        let v: Vec<serde_json::Value> = serde_json::from_slice(&resp.body).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid JSON: {e}"),
            )
        })?;
        Ok(v)
    }

    pub fn register(
        &self,
        name: &str,
        port: u16,
        tags: Option<&HashMap<String, String>>,
        check_script: Option<&str>,
        host: Option<&str>,
    ) -> Result<(), ConsulError> {
        let id = format!("{name}-{port}");
        let mut payload = serde_json::json!({
            "id": id,
            "name": name,
            "port": port as i64,
            "check": {"ttl": "60s"},
        });

        if let Some(tags) = tags {
            let arr: Vec<String> = tags.iter().map(|(k, v)| format!("{k}:{v}")).collect();
            payload["tags"] = serde_json::json!(arr);
        }
        if let Some(script) = check_script {
            payload["check"] = serde_json::json!({"interval": "30s", "script": script});
        }

        let host = host.unwrap_or(&self.consul_host).to_string();
        let body = serde_json::to_vec(&payload).map_err(|e| ConsulError(e.to_string()))?;
        let resp = self
            .http
            .request(
                "PUT",
                &host,
                self.consul_port,
                "/v1/agent/service/register",
                Some(&body),
                Duration::from_secs(15),
            )
            .map_err(|e| ConsulError(e.to_string()))?;
        if resp.status != 200 {
            return Err(ConsulError(String::from_utf8_lossy(&resp.body).to_string()));
        }

        // Best-effort keepalive thread for TTL health checks. Detach like Python's daemon.
        let http = Arc::clone(&self.http);
        let consul_port = self.consul_port;
        std::thread::spawn(move || loop {
            // Sleep roughly 30s between pings; on failures, retry sooner.
            let start = now_secs_f64();
            let path = format!("/v1/agent/check/pass/service:{id}");
            let _ = http.request(
                "GET",
                &host,
                consul_port,
                &path,
                None,
                Duration::from_secs(5),
            );
            let elapsed = now_secs_f64() - start;
            let sleep = (30.0 - elapsed).max(0.0);
            std::thread::sleep(Duration::from_secs_f64(sleep));
        });

        Ok(())
    }

    pub fn deregister(&self, name: &str, port: u16, host: Option<&str>) -> Result<(), ConsulError> {
        let host = host.unwrap_or(&self.consul_host);
        let path = format!("/v1/agent/service/deregister/{name}-{port}");
        let resp = self
            .http
            .request(
                "PUT",
                host,
                self.consul_port,
                &path,
                None,
                Duration::from_secs(15),
            )
            .map_err(|e| ConsulError(e.to_string()))?;
        if resp.status != 200 {
            return Err(ConsulError(String::from_utf8_lossy(&resp.body).to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct MockHttp {
        status: u16,
        body: Vec<u8>,
        calls: AtomicUsize,
    }

    impl HttpTransport for MockHttp {
        fn request(
            &self,
            _method: &str,
            _host: &str,
            _port: u16,
            _path: &str,
            _body: Option<&[u8]>,
            _timeout: Duration,
        ) -> std::io::Result<HttpResponse> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(HttpResponse {
                status: self.status,
                body: self.body.clone(),
            })
        }
    }

    #[test]
    fn test_lookup_matches_python() {
        let data =
            serde_json::json!([{"Port": 1234, "Host": "192.168.0.1", "Tags": {"index": "0"}}]);
        let body = serde_json::to_vec(&data).expect("serializing lookup fixture JSON should succeed");
        let http = Arc::new(MockHttp {
            status: 200,
            body,
            calls: AtomicUsize::new(0),
        });
        let client = Client::with_http(http);
        let v = client
            .lookup("test_name", 3, 0)
            .expect("lookup should succeed with mock HTTP 200 response");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["Port"], 1234);
    }

    #[test]
    fn test_register_ok() {
        let http = Arc::new(MockHttp {
            status: 200,
            body: Vec::new(),
            calls: AtomicUsize::new(0),
        });
        let client = Client::with_http(http);
        client
            .register("test_name", 12345, None, None, None)
            .expect("register should succeed with mock HTTP 200 response");
    }

    #[test]
    fn test_deregister_ok() {
        let http = Arc::new(MockHttp {
            status: 200,
            body: Vec::new(),
            calls: AtomicUsize::new(0),
        });
        let client = Client::with_http(http);
        client
            .deregister("test_name", 12345, None)
            .expect("deregister should succeed with mock HTTP 200 response");
    }
}
