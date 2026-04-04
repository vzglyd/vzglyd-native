//! VZGLYD Native Binary
//!
//! Main entry point for the native host.

use std::path::Path;
use std::process::ExitCode;

use vzglyd_native::NativeApp;
use vzglyd_native::app::RunConfig;
use vzglyd_native::slide_loader;

const DEFAULT_SLIDES_DIR: &str = "slides";

#[derive(Debug)]
enum Command {
    Run(RunConfig),
    Pack {
        source_dir: String,
        output_path: String,
        verbose: bool,
    },
}

fn main() -> ExitCode {
    let command = match parse_command() {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    init_logging(command_verbose(&command));

    match command {
        Command::Pack {
            source_dir,
            output_path,
            ..
        } => match slide_loader::pack_slide_directory(
            Path::new(&source_dir),
            Path::new(&output_path),
        ) {
            Ok(report) => {
                eprintln!(
                    "[vzglyd] packed {} -> {} ({:.1}% overhead)",
                    source_dir,
                    report.output_path.display(),
                    report.overhead_ratio() * 100.0
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                log::error!("pack failed: {error}");
                ExitCode::FAILURE
            }
        },
        Command::Run(run_config) => match NativeApp::run(run_config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                log::error!("application error: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

fn parse_command() -> Result<Command, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "pack") {
        return parse_pack_command(&args[2..]);
    }
    parse_run_command(&args[1..]).map(Command::Run)
}

fn parse_pack_command(args: &[String]) -> Result<Command, String> {
    let source_dir = args
        .first()
        .ok_or_else(|| "usage: vzglyd pack <slide-dir> -o <archive.vzglyd>".to_string())?;

    let mut output_path = None;
    let mut verbose = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                let Some(path) = args.get(i + 1) else {
                    return Err("missing output path after -o".into());
                };
                output_path = Some(path.clone());
                i += 2;
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "unknown pack argument '{other}'; usage: vzglyd pack <slide-dir> -o <archive.vzglyd>"
                ));
            }
        }
    }

    let output_path = output_path.ok_or_else(|| {
        "missing -o <archive.vzglyd>; usage: vzglyd pack <slide-dir> -o <archive.vzglyd>"
            .to_string()
    })?;

    Ok(Command::Pack {
        source_dir: source_dir.to_string(),
        output_path,
        verbose,
    })
}

fn parse_run_command(args: &[String]) -> Result<RunConfig, String> {
    let mut slides_dir = Some(DEFAULT_SLIDES_DIR.to_string());
    let mut scene_path = None;
    let mut trace_session = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => {
                i += 1;
            }
            "-d" | "--slides-dir" => {
                let Some(dir) = args.get(i + 1) else {
                    return Err("missing path after --slides-dir".into());
                };
                slides_dir = Some(dir.clone());
                scene_path = None;
                i += 2;
            }
            "--scene" => {
                let Some(path) = args.get(i + 1) else {
                    return Err("missing path after --scene".into());
                };
                scene_path = Some(path.clone());
                slides_dir = None;
                i += 2;
            }
            "--trace-session" => {
                let Some(path) = args.get(i + 1) else {
                    return Err("missing path after --trace-session".into());
                };
                trace_session = Some(path.clone());
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument '{other}'"));
            }
        }
    }

    Ok(RunConfig {
        slides_dir,
        scene_path,
        trace_session,
    })
}

fn command_verbose(command: &Command) -> bool {
    match command {
        Command::Run(_) => std::env::args().any(|arg| arg == "-v" || arg == "--verbose"),
        Command::Pack { verbose, .. } => *verbose,
    }
}

fn init_logging(verbose: bool) {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(if verbose {
            "info"
        } else {
            "warn"
        }));
    builder.filter_module("wgpu_hal", log::LevelFilter::Off);
    builder.filter_module("wgpu_core", log::LevelFilter::Off);
    builder.init();
}

fn print_help() {
    println!("VZGLYD Native Host");
    println!();
    println!("Usage:");
    println!("  vzglyd [--slides-dir <DIR> | --scene <PATH>] [--verbose]");
    println!("  vzglyd pack <slide-dir> -o <archive.vzglyd> [--verbose]");
    println!();
    println!("Options:");
    println!("  -d, --slides-dir <DIR>  Shared slides repo root (expects playlist.json)");
    println!("      --scene <PATH>      Run a single slide package directly");
    println!("      --trace-session <DIR>  Write Perfetto trace output into DIR");
    println!("  -v, --verbose           Enable verbose logging");
    println!("  -h, --help              Print this help message");
    println!();
    println!(
        "When no slide source is provided, the host looks for a shared slides repo at '{}'.",
        DEFAULT_SLIDES_DIR
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_command_defaults_to_standard_slides_dir() {
        let run = parse_run_command(&[]).expect("parse run");
        assert_eq!(run.slides_dir.as_deref(), Some(DEFAULT_SLIDES_DIR));
        assert_eq!(run.scene_path, None);
    }

    #[test]
    fn run_command_prefers_single_scene() {
        let run = parse_run_command(&["--scene".into(), "demo.vzglyd".into()]).expect("parse");
        assert_eq!(run.scene_path.as_deref(), Some("demo.vzglyd"));
        assert_eq!(run.slides_dir, None);
    }

    #[test]
    fn pack_command_requires_output() {
        let error = parse_pack_command(&["slides/demo".into()]).expect_err("missing output");
        assert!(error.contains("missing -o"));
    }
}
