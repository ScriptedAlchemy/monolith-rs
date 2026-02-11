use clap::Parser;
use monolith_cli::Cli;

#[test]
fn cli_parses_agent_service_controller_flags() {
    let _cli = Cli::parse_from([
        "monolith",
        "agent-service",
        "controller",
        "--fake-zk",
        "--bzid",
        "gip",
        "--cmd",
        "bzid-info",
    ]);
}
