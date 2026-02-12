//! Kafka data source for streaming data into Monolith pipelines.
//!
//! This module provides [`KafkaDataSource`] for consuming messages from Apache Kafka
//! and converting them to [`Example`] protobuf messages that can be processed by
//! the Monolith data pipeline.
//!
//! # Feature Flag
//!
//! This module requires the `kafka` feature to be enabled. When the feature is not
//! enabled, a stub implementation is provided that returns appropriate errors.
//!
//! ```toml
//! [dependencies]
//! monolith-data = { version = "0.1", features = ["kafka"] }
//! ```
//!
//! # Architecture
//!
//! The Kafka data source provides a streaming interface for consuming messages:
//!
//! - [`KafkaConfig`] - Configuration for connecting to Kafka
//! - [`KafkaDataSource`] - The main consumer interface
//! - [`KafkaConsumer`] - Trait for consumer implementations
//! - [`MockKafkaConsumer`] - Mock implementation for testing
//!
//! # Example
//!
//! ```no_run
//! use monolith_data::kafka::{KafkaConfig, KafkaDataSource, OffsetReset};
//!
//! let config = KafkaConfig::new(
//!     vec!["localhost:9092".to_string()],
//!     "my-topic".to_string(),
//!     "my-consumer-group".to_string(),
//! );
//!
//! let mut source = KafkaDataSource::new(config).unwrap();
//! source.subscribe(&["my-topic"]).unwrap();
//!
//! // Poll for messages
//! while let Some(message) = source.poll(std::time::Duration::from_millis(100)) {
//!     println!("Received: {:?}", message);
//! }
//! ```

use monolith_proto::Example;
use prost::Message as ProstMessage;
use std::collections::VecDeque;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur when working with Kafka.
#[derive(Error, Debug)]
pub enum KafkaError {
    /// Kafka feature is not enabled.
    #[error("Kafka support is not enabled. Enable the 'kafka' feature flag.")]
    FeatureNotEnabled,

    /// Failed to connect to Kafka brokers.
    #[error("Failed to connect to Kafka brokers: {0}")]
    ConnectionError(String),

    /// Failed to subscribe to topic(s).
    #[error("Failed to subscribe to topic(s): {0}")]
    SubscriptionError(String),

    /// Failed to commit offsets.
    #[error("Failed to commit offsets: {0}")]
    CommitError(String),

    /// Failed to poll for messages.
    #[error("Failed to poll for messages: {0}")]
    PollError(String),

    /// Failed to decode message payload.
    #[error("Failed to decode message: {0}")]
    DecodeError(#[from] prost::DecodeError),

    /// Consumer is not subscribed to any topics.
    #[error("Consumer is not subscribed to any topics")]
    NotSubscribed,

    /// Consumer has already been closed.
    #[error("Consumer has already been closed")]
    ConsumerClosed,

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for Kafka operations.
pub type Result<T> = std::result::Result<T, KafkaError>;

/// Offset reset policy for Kafka consumers.
///
/// Determines where to start consuming when there is no initial offset
/// or the current offset no longer exists on the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffsetReset {
    /// Start consuming from the earliest available offset.
    #[default]
    Earliest,
    /// Start consuming from the latest offset (only new messages).
    Latest,
    /// Fail if no offset is available.
    None,
}

impl OffsetReset {
    /// Returns the string representation used by Kafka.
    pub fn as_str(&self) -> &'static str {
        match self {
            OffsetReset::Earliest => "earliest",
            OffsetReset::Latest => "latest",
            OffsetReset::None => "none",
        }
    }
}

impl std::fmt::Display for OffsetReset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Configuration for connecting to Kafka.
///
/// # Example
///
/// ```
/// use monolith_data::kafka::{KafkaConfig, OffsetReset};
///
/// let config = KafkaConfig::new(
///     vec!["broker1:9092".to_string(), "broker2:9092".to_string()],
///     "my-topic".to_string(),
///     "my-consumer-group".to_string(),
/// )
/// .with_offset_reset(OffsetReset::Latest)
/// .with_max_poll_records(500);
/// ```
#[derive(Debug, Clone)]
pub struct KafkaConfig {
    /// List of Kafka broker addresses (host:port).
    pub brokers: Vec<String>,
    /// The topic to consume from.
    pub topic: String,
    /// Consumer group ID for offset management.
    pub group_id: String,
    /// Offset reset policy.
    pub offset_reset: OffsetReset,
    /// Maximum number of records to return in a single poll.
    pub max_poll_records: usize,
}

impl KafkaConfig {
    /// Creates a new Kafka configuration with default settings.
    ///
    /// # Arguments
    ///
    /// * `brokers` - List of Kafka broker addresses
    /// * `topic` - The topic to consume from
    /// * `group_id` - Consumer group ID
    pub fn new(brokers: Vec<String>, topic: String, group_id: String) -> Self {
        Self {
            brokers,
            topic,
            group_id,
            offset_reset: OffsetReset::default(),
            max_poll_records: 500,
        }
    }

    /// Sets the offset reset policy.
    pub fn with_offset_reset(mut self, offset_reset: OffsetReset) -> Self {
        self.offset_reset = offset_reset;
        self
    }

    /// Sets the maximum number of records to return per poll.
    pub fn with_max_poll_records(mut self, max_poll_records: usize) -> Self {
        self.max_poll_records = max_poll_records;
        self
    }

    /// Returns the broker list as a comma-separated string.
    pub fn broker_string(&self) -> String {
        self.brokers.join(",")
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.brokers.is_empty() {
            return Err(KafkaError::ConfigError(
                "At least one broker must be specified".to_string(),
            ));
        }
        if self.topic.is_empty() {
            return Err(KafkaError::ConfigError(
                "Topic must not be empty".to_string(),
            ));
        }
        if self.group_id.is_empty() {
            return Err(KafkaError::ConfigError(
                "Group ID must not be empty".to_string(),
            ));
        }
        if self.max_poll_records == 0 {
            return Err(KafkaError::ConfigError(
                "max_poll_records must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// A message consumed from Kafka.
#[derive(Debug, Clone)]
pub struct KafkaMessage {
    /// The topic this message came from.
    pub topic: String,
    /// The partition this message came from.
    pub partition: i32,
    /// The offset of this message within the partition.
    pub offset: i64,
    /// The message key (optional).
    pub key: Option<Vec<u8>>,
    /// The message payload.
    pub payload: Vec<u8>,
    /// The timestamp of the message (milliseconds since epoch).
    pub timestamp: Option<i64>,
}

impl KafkaMessage {
    /// Attempts to decode the payload as an Example protobuf.
    ///
    /// # Returns
    ///
    /// The decoded Example, or an error if decoding fails.
    pub fn decode_example(&self) -> Result<Example> {
        Example::decode(self.payload.as_slice()).map_err(KafkaError::from)
    }
}

/// Trait for Kafka consumer implementations.
///
/// This trait abstracts the Kafka consumer interface, allowing for different
/// implementations (real rdkafka, mock for testing, etc.).
pub trait KafkaConsumer: Send {
    /// Subscribes to the specified topics.
    ///
    /// # Arguments
    ///
    /// * `topics` - List of topic names to subscribe to
    fn subscribe(&mut self, topics: &[&str]) -> Result<()>;

    /// Polls for the next message.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for a message
    ///
    /// # Returns
    ///
    /// The next message if available within the timeout, or `None`.
    fn poll(&mut self, timeout: Duration) -> Option<KafkaMessage>;

    /// Commits the current offsets synchronously.
    fn commit(&mut self) -> Result<()>;

    /// Closes the consumer, releasing resources.
    fn close(&mut self);

    /// Returns whether the consumer is closed.
    fn is_closed(&self) -> bool;
}

/// Mock Kafka consumer for testing.
///
/// This implementation stores messages in memory and allows tests to
/// control what messages are returned.
///
/// # Example
///
/// ```
/// use monolith_data::kafka::{MockKafkaConsumer, KafkaMessage, KafkaConsumer};
/// use std::time::Duration;
///
/// let mut consumer = MockKafkaConsumer::new();
///
/// // Add test messages
/// consumer.add_message(KafkaMessage {
///     topic: "test-topic".to_string(),
///     partition: 0,
///     offset: 0,
///     key: None,
///     payload: vec![1, 2, 3],
///     timestamp: Some(1234567890),
/// });
///
/// consumer.subscribe(&["test-topic"]).unwrap();
///
/// // Poll returns the added message
/// let msg = consumer.poll(Duration::from_millis(100)).unwrap();
/// assert_eq!(msg.offset, 0);
/// ```
#[derive(Debug, Default)]
pub struct MockKafkaConsumer {
    messages: VecDeque<KafkaMessage>,
    subscribed_topics: Vec<String>,
    committed_offsets: Vec<(String, i32, i64)>,
    closed: bool,
}

impl MockKafkaConsumer {
    /// Creates a new mock consumer with no messages.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mock consumer pre-populated with messages.
    pub fn with_messages(messages: Vec<KafkaMessage>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            subscribed_topics: Vec::new(),
            committed_offsets: Vec::new(),
            closed: false,
        }
    }

    /// Adds a message to the consumer's queue.
    pub fn add_message(&mut self, message: KafkaMessage) {
        self.messages.push_back(message);
    }

    /// Adds multiple messages to the consumer's queue.
    pub fn add_messages(&mut self, messages: impl IntoIterator<Item = KafkaMessage>) {
        self.messages.extend(messages);
    }

    /// Returns the topics this consumer is subscribed to.
    pub fn subscribed_topics(&self) -> &[String] {
        &self.subscribed_topics
    }

    /// Returns the committed offsets as (topic, partition, offset) tuples.
    pub fn committed_offsets(&self) -> &[(String, i32, i64)] {
        &self.committed_offsets
    }

    /// Returns the number of remaining messages in the queue.
    pub fn remaining_messages(&self) -> usize {
        self.messages.len()
    }
}

impl KafkaConsumer for MockKafkaConsumer {
    fn subscribe(&mut self, topics: &[&str]) -> Result<()> {
        if self.closed {
            return Err(KafkaError::ConsumerClosed);
        }
        self.subscribed_topics = topics.iter().map(|s| s.to_string()).collect();
        Ok(())
    }

    fn poll(&mut self, _timeout: Duration) -> Option<KafkaMessage> {
        if self.closed || self.subscribed_topics.is_empty() {
            return None;
        }
        // Only return messages for subscribed topics
        let mut idx = None;
        for (i, msg) in self.messages.iter().enumerate() {
            if self.subscribed_topics.contains(&msg.topic) {
                idx = Some(i);
                break;
            }
        }
        idx.and_then(|i| self.messages.remove(i))
    }

    fn commit(&mut self) -> Result<()> {
        if self.closed {
            return Err(KafkaError::ConsumerClosed);
        }
        // In a real implementation, this would commit offsets to Kafka
        Ok(())
    }

    fn close(&mut self) {
        self.closed = true;
        self.messages.clear();
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Kafka data source for streaming Example protos from Kafka.
///
/// This struct wraps a [`KafkaConsumer`] implementation and provides
/// a convenient interface for consuming messages and converting them
/// to Example protobuf messages.
///
/// # Feature Flag
///
/// When the `kafka` feature is enabled, this can be constructed with
/// a real Kafka consumer (using rdkafka). When the feature is disabled,
/// the [`new`](Self::new) method returns an error, but you can still
/// use [`with_consumer`](Self::with_consumer) with a mock consumer for testing.
pub struct KafkaDataSource<C: KafkaConsumer = MockKafkaConsumer> {
    consumer: C,
    config: KafkaConfig,
}

impl KafkaDataSource<MockKafkaConsumer> {
    /// Creates a new Kafka data source.
    ///
    /// # Feature Flag
    ///
    /// When the `kafka` feature is not enabled, this returns a
    /// [`KafkaError::FeatureNotEnabled`] error. Use [`with_mock_consumer`](Self::with_mock_consumer)
    /// for testing without the kafka feature.
    ///
    /// # Arguments
    ///
    /// * `config` - The Kafka configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `kafka` feature is not enabled
    /// - The configuration is invalid
    /// - Connection to Kafka fails
    #[cfg(not(feature = "kafka"))]
    pub fn new(config: KafkaConfig) -> Result<Self> {
        config.validate()?;
        Err(KafkaError::FeatureNotEnabled)
    }

    /// Creates a new Kafka data source with the kafka feature enabled.
    ///
    /// # Note
    ///
    /// This is a placeholder for the real rdkafka implementation.
    /// In production, this would create an actual Kafka consumer.
    #[cfg(feature = "kafka")]
    pub fn new(config: KafkaConfig) -> Result<Self> {
        config.validate()?;
        // In a real implementation, this would create an rdkafka consumer.
        // For now, we create a mock consumer as a placeholder.
        Ok(Self {
            consumer: MockKafkaConsumer::new(),
            config,
        })
    }

    /// Creates a new Kafka data source with a mock consumer for testing.
    ///
    /// This method works regardless of whether the `kafka` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `config` - The Kafka configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn with_mock_consumer(config: KafkaConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            consumer: MockKafkaConsumer::new(),
            config,
        })
    }
}

impl<C: KafkaConsumer> KafkaDataSource<C> {
    /// Creates a new Kafka data source with a custom consumer implementation.
    ///
    /// This allows injecting custom consumer implementations for testing
    /// or using alternative Kafka client libraries.
    ///
    /// # Arguments
    ///
    /// * `config` - The Kafka configuration
    /// * `consumer` - The consumer implementation
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn with_consumer(config: KafkaConfig, consumer: C) -> Result<Self> {
        config.validate()?;
        Ok(Self { consumer, config })
    }

    /// Returns the configuration for this data source.
    pub fn config(&self) -> &KafkaConfig {
        &self.config
    }

    /// Subscribes to the specified topics.
    ///
    /// # Arguments
    ///
    /// * `topics` - List of topic names to subscribe to
    ///
    /// # Errors
    ///
    /// Returns an error if subscription fails.
    pub fn subscribe(&mut self, topics: &[&str]) -> Result<()> {
        self.consumer.subscribe(topics)
    }

    /// Subscribes to the configured topic.
    ///
    /// This is a convenience method that subscribes to the topic
    /// specified in the configuration.
    pub fn subscribe_configured(&mut self) -> Result<()> {
        let topic = self.config.topic.clone();
        self.subscribe(&[&topic])
    }

    /// Polls for the next message.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for a message
    ///
    /// # Returns
    ///
    /// The next message if available within the timeout, or `None`.
    pub fn poll(&mut self, timeout: Duration) -> Option<KafkaMessage> {
        self.consumer.poll(timeout)
    }

    /// Polls for the next message and decodes it as an Example.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for a message
    ///
    /// # Returns
    ///
    /// The decoded Example if a message is available, or `None`.
    /// Returns an error if decoding fails.
    pub fn poll_example(&mut self, timeout: Duration) -> Option<Result<Example>> {
        self.poll(timeout).map(|msg| msg.decode_example())
    }

    /// Commits the current offsets synchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the commit fails.
    pub fn commit(&mut self) -> Result<()> {
        self.consumer.commit()
    }

    /// Closes the consumer and releases resources.
    pub fn close(&mut self) {
        self.consumer.close();
    }

    /// Returns whether the consumer is closed.
    pub fn is_closed(&self) -> bool {
        self.consumer.is_closed()
    }

    /// Returns a mutable reference to the underlying consumer.
    ///
    /// This is primarily useful for testing with mock consumers.
    pub fn consumer_mut(&mut self) -> &mut C {
        &mut self.consumer
    }
}

/// Iterator adapter for consuming Examples from Kafka.
///
/// This iterator polls Kafka for messages and decodes them as Examples.
/// It continues polling until the consumer is closed or an error occurs.
pub struct KafkaExampleIterator<'a, C: KafkaConsumer> {
    source: &'a mut KafkaDataSource<C>,
    timeout: Duration,
    stop_on_error: bool,
}

impl<'a, C: KafkaConsumer> KafkaExampleIterator<'a, C> {
    /// Creates a new iterator over Examples from the Kafka source.
    pub fn new(source: &'a mut KafkaDataSource<C>, timeout: Duration) -> Self {
        Self {
            source,
            timeout,
            stop_on_error: false,
        }
    }

    /// Configures the iterator to stop when a decode error occurs.
    pub fn stop_on_error(mut self) -> Self {
        self.stop_on_error = true;
        self
    }
}

impl<'a, C: KafkaConsumer> Iterator for KafkaExampleIterator<'a, C> {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.source.poll_example(self.timeout) {
                Some(Ok(example)) => return Some(example),
                Some(Err(_)) if self.stop_on_error => return None,
                Some(Err(_)) => continue, // Skip decode errors
                None => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::{add_feature, create_example};

    fn make_test_config() -> KafkaConfig {
        KafkaConfig::new(
            vec!["localhost:9092".to_string()],
            "test-topic".to_string(),
            "test-group".to_string(),
        )
    }

    fn make_example_message(offset: i64) -> KafkaMessage {
        let mut example = create_example();
        add_feature(&mut example, "id", vec![offset], vec![offset as f32]);
        let payload = example.encode_to_vec();

        KafkaMessage {
            topic: "test-topic".to_string(),
            partition: 0,
            offset,
            key: None,
            payload,
            timestamp: Some(1234567890 + offset),
        }
    }

    #[test]
    fn test_offset_reset_as_str() {
        assert_eq!(OffsetReset::Earliest.as_str(), "earliest");
        assert_eq!(OffsetReset::Latest.as_str(), "latest");
        assert_eq!(OffsetReset::None.as_str(), "none");
    }

    #[test]
    fn test_offset_reset_display() {
        assert_eq!(format!("{}", OffsetReset::Earliest), "earliest");
        assert_eq!(format!("{}", OffsetReset::Latest), "latest");
        assert_eq!(format!("{}", OffsetReset::None), "none");
    }

    #[test]
    fn test_kafka_config_new() {
        let config = KafkaConfig::new(
            vec!["broker1:9092".to_string(), "broker2:9092".to_string()],
            "my-topic".to_string(),
            "my-group".to_string(),
        );

        assert_eq!(config.brokers.len(), 2);
        assert_eq!(config.topic, "my-topic");
        assert_eq!(config.group_id, "my-group");
        assert_eq!(config.offset_reset, OffsetReset::Earliest);
        assert_eq!(config.max_poll_records, 500);
    }

    #[test]
    fn test_kafka_config_builder() {
        let config = KafkaConfig::new(
            vec!["localhost:9092".to_string()],
            "topic".to_string(),
            "group".to_string(),
        )
        .with_offset_reset(OffsetReset::Latest)
        .with_max_poll_records(1000);

        assert_eq!(config.offset_reset, OffsetReset::Latest);
        assert_eq!(config.max_poll_records, 1000);
    }

    #[test]
    fn test_kafka_config_broker_string() {
        let config = KafkaConfig::new(
            vec![
                "broker1:9092".to_string(),
                "broker2:9092".to_string(),
                "broker3:9092".to_string(),
            ],
            "topic".to_string(),
            "group".to_string(),
        );

        assert_eq!(
            config.broker_string(),
            "broker1:9092,broker2:9092,broker3:9092"
        );
    }

    #[test]
    fn test_kafka_config_validate() {
        // Valid config
        let config = make_test_config();
        config
            .validate()
            .expect("default kafka test config should pass validation");

        // Empty brokers
        let config = KafkaConfig::new(vec![], "topic".to_string(), "group".to_string());
        assert!(matches!(config.validate(), Err(KafkaError::ConfigError(_))));

        // Empty topic
        let config = KafkaConfig::new(
            vec!["localhost:9092".to_string()],
            "".to_string(),
            "group".to_string(),
        );
        assert!(matches!(config.validate(), Err(KafkaError::ConfigError(_))));

        // Empty group_id
        let config = KafkaConfig::new(
            vec!["localhost:9092".to_string()],
            "topic".to_string(),
            "".to_string(),
        );
        assert!(matches!(config.validate(), Err(KafkaError::ConfigError(_))));

        // Zero max_poll_records
        let config = KafkaConfig::new(
            vec!["localhost:9092".to_string()],
            "topic".to_string(),
            "group".to_string(),
        )
        .with_max_poll_records(0);
        assert!(matches!(config.validate(), Err(KafkaError::ConfigError(_))));
    }

    #[test]
    fn test_kafka_message_decode_example() {
        let message = make_example_message(42);
        let example = message.decode_example().unwrap();

        let feature = example
            .named_feature
            .iter()
            .find(|nf| nf.name == "id")
            .unwrap();
        let data = crate::example::extract_feature_data(feature.feature.as_ref().unwrap());
        assert_eq!(data.fid, vec![42]);
    }

    #[test]
    fn test_kafka_message_decode_invalid() {
        let message = KafkaMessage {
            topic: "test-topic".to_string(),
            partition: 0,
            offset: 0,
            key: None,
            payload: vec![0xff, 0xff, 0xff], // Invalid protobuf
            timestamp: None,
        };

        assert!(matches!(
            message.decode_example(),
            Err(KafkaError::DecodeError(_))
        ));
    }

    #[test]
    fn test_mock_consumer_subscribe() {
        let mut consumer = MockKafkaConsumer::new();
        consumer.subscribe(&["topic1", "topic2"]).unwrap();

        assert_eq!(consumer.subscribed_topics(), &["topic1", "topic2"]);
    }

    #[test]
    fn test_mock_consumer_poll() {
        let mut consumer = MockKafkaConsumer::new();

        // Add messages
        consumer.add_message(make_example_message(0));
        consumer.add_message(make_example_message(1));

        // Must subscribe first
        assert!(consumer.poll(Duration::from_millis(100)).is_none());

        consumer.subscribe(&["test-topic"]).unwrap();

        let msg = consumer.poll(Duration::from_millis(100)).unwrap();
        assert_eq!(msg.offset, 0);

        let msg = consumer.poll(Duration::from_millis(100)).unwrap();
        assert_eq!(msg.offset, 1);

        // No more messages
        assert!(consumer.poll(Duration::from_millis(100)).is_none());
    }

    #[test]
    fn test_mock_consumer_poll_filters_topics() {
        let mut consumer = MockKafkaConsumer::new();

        consumer.add_message(KafkaMessage {
            topic: "topic1".to_string(),
            partition: 0,
            offset: 0,
            key: None,
            payload: vec![],
            timestamp: None,
        });
        consumer.add_message(KafkaMessage {
            topic: "topic2".to_string(),
            partition: 0,
            offset: 1,
            key: None,
            payload: vec![],
            timestamp: None,
        });

        consumer.subscribe(&["topic2"]).unwrap();

        let msg = consumer.poll(Duration::from_millis(100)).unwrap();
        assert_eq!(msg.topic, "topic2");
        assert_eq!(msg.offset, 1);
    }

    #[test]
    fn test_mock_consumer_close() {
        let mut consumer = MockKafkaConsumer::new();
        consumer.add_message(make_example_message(0));
        consumer.subscribe(&["test-topic"]).unwrap();

        assert!(!consumer.is_closed());

        consumer.close();

        assert!(consumer.is_closed());
        assert!(consumer.poll(Duration::from_millis(100)).is_none());
        assert!(matches!(
            consumer.subscribe(&["test-topic"]),
            Err(KafkaError::ConsumerClosed)
        ));
        assert!(matches!(consumer.commit(), Err(KafkaError::ConsumerClosed)));
    }

    #[test]
    fn test_mock_consumer_with_messages() {
        let messages = vec![make_example_message(0), make_example_message(1)];
        let consumer = MockKafkaConsumer::with_messages(messages);

        assert_eq!(consumer.remaining_messages(), 2);
    }

    #[test]
    fn test_kafka_data_source_with_mock_consumer() {
        let config = make_test_config();
        let mut source = KafkaDataSource::with_mock_consumer(config).unwrap();

        // Add test messages
        source.consumer_mut().add_message(make_example_message(0));
        source.consumer_mut().add_message(make_example_message(1));

        source.subscribe_configured().unwrap();

        // Poll for messages
        let msg = source.poll(Duration::from_millis(100)).unwrap();
        assert_eq!(msg.offset, 0);

        // Commit should succeed
        source.commit().unwrap();

        // Close
        source.close();
        assert!(source.is_closed());
    }

    #[test]
    fn test_kafka_data_source_poll_example() {
        let config = make_test_config();
        let mut source = KafkaDataSource::with_mock_consumer(config).unwrap();

        source.consumer_mut().add_message(make_example_message(42));
        source.subscribe_configured().unwrap();

        let example = source
            .poll_example(Duration::from_millis(100))
            .unwrap()
            .unwrap();
        let feature = example
            .named_feature
            .iter()
            .find(|nf| nf.name == "id")
            .unwrap();
        let data = crate::example::extract_feature_data(feature.feature.as_ref().unwrap());
        assert_eq!(data.fid, vec![42]);
    }

    #[test]
    fn test_kafka_data_source_with_custom_consumer() {
        let config = make_test_config();
        let consumer = MockKafkaConsumer::with_messages(vec![make_example_message(0)]);

        let mut source = KafkaDataSource::with_consumer(config, consumer).unwrap();
        source.subscribe_configured().unwrap();

        assert!(source.poll(Duration::from_millis(100)).is_some());
    }

    #[test]
    fn test_kafka_data_source_config() {
        let config = make_test_config();
        let source = KafkaDataSource::with_mock_consumer(config.clone()).unwrap();

        assert_eq!(source.config().topic, config.topic);
        assert_eq!(source.config().group_id, config.group_id);
    }

    #[test]
    #[cfg(not(feature = "kafka"))]
    fn test_kafka_data_source_new_without_feature() {
        let config = make_test_config();
        let result = KafkaDataSource::new(config);

        assert!(matches!(result, Err(KafkaError::FeatureNotEnabled)));
    }

    #[test]
    fn test_kafka_example_iterator() {
        let config = make_test_config();
        let mut source = KafkaDataSource::with_mock_consumer(config).unwrap();

        // Add messages
        for i in 0..5 {
            source.consumer_mut().add_message(make_example_message(i));
        }
        source.subscribe_configured().unwrap();

        // Create iterator and collect examples
        let iter = KafkaExampleIterator::new(&mut source, Duration::from_millis(100));
        let examples: Vec<_> = iter.collect();

        assert_eq!(examples.len(), 5);
    }

    #[test]
    fn test_kafka_example_iterator_skip_errors() {
        let config = make_test_config();
        let mut source = KafkaDataSource::with_mock_consumer(config).unwrap();

        // Add a valid message, an invalid message, and another valid message
        source.consumer_mut().add_message(make_example_message(0));
        source.consumer_mut().add_message(KafkaMessage {
            topic: "test-topic".to_string(),
            partition: 0,
            offset: 1,
            key: None,
            payload: vec![0xff, 0xff], // Invalid
            timestamp: None,
        });
        source.consumer_mut().add_message(make_example_message(2));

        source.subscribe_configured().unwrap();

        let iter = KafkaExampleIterator::new(&mut source, Duration::from_millis(100));
        let examples: Vec<_> = iter.collect();

        // Should skip the invalid message
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_kafka_example_iterator_stop_on_error() {
        let config = make_test_config();
        let mut source = KafkaDataSource::with_mock_consumer(config).unwrap();

        source.consumer_mut().add_message(make_example_message(0));
        source.consumer_mut().add_message(KafkaMessage {
            topic: "test-topic".to_string(),
            partition: 0,
            offset: 1,
            key: None,
            payload: vec![0xff, 0xff], // Invalid
            timestamp: None,
        });
        source.consumer_mut().add_message(make_example_message(2));

        source.subscribe_configured().unwrap();

        let iter =
            KafkaExampleIterator::new(&mut source, Duration::from_millis(100)).stop_on_error();
        let examples: Vec<_> = iter.collect();

        // Should stop at the error
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_kafka_error_display() {
        let err = KafkaError::FeatureNotEnabled;
        assert!(err.to_string().contains("not enabled"));

        let err = KafkaError::ConnectionError("timeout".to_string());
        assert!(err.to_string().contains("timeout"));

        let err = KafkaError::ConsumerClosed;
        assert!(err.to_string().contains("closed"));
    }
}
