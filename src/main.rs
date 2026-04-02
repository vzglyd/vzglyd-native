//! VZGLYD Native Binary
//!
//! Main entry point for the native host.

use std::process::ExitCode;
use vzglyd_native::NativeApp;

fn main() -> ExitCode {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let slides_dir = parse_args(&args);

    // Initialize logging
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(if verbose {
            "info"
        } else {
            "warn"
        }));
    builder.filter_module("wgpu_hal", log::LevelFilter::Off);
    builder.filter_module("wgpu_core", log::LevelFilter::Off);
    builder.init();

    eprintln!(
        "[vzglyd] starting (debug build — WASM compilation will be slow; prefer cargo run --release)"
    );
    if let Some(ref dir) = slides_dir {
        eprintln!("[vzglyd] slides dir: {}", dir);
    } else {
        eprintln!("[vzglyd] no --slides-dir specified");
    }

    // Run the application
    match NativeApp::run(slides_dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("Application error: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Parses command line arguments.
fn parse_args(args: &[String]) -> Option<String> {
    let mut i = 1;
    let mut slides_dir = None;

    while i < args.len() {
        match args[i].as_str() {
            "--slides-dir" | "-d" => {
                if let Some(dir) = args.get(i + 1) {
                    slides_dir = Some(dir.clone());
                    i += 2;
                    continue;
                }
            }
            "--verbose" | "-v" => {
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    slides_dir
}

/// Prints help message.
fn print_help() {
    println!("VZGLYD Native Host - WebAssembly slide display engine");
    println!();
    println!("Usage: vzglyd [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -d, --slides-dir <DIR>  Directory containing .wasm slide files");
    println!("  -v, --verbose           Enable verbose logging");
    println!("  -h, --help              Print this help message");
    println!();
    println!("The application will display slides in alphabetical order with");
    println!("smooth transitions between them.");
}
