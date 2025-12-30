//! Common utilities for the compiler.

#![feature(const_trait_impl)]

pub mod span;

/// Compilation target.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// Compile to a native binary for the current platform with LLVM. Default for the `release`
    /// profile.
    Native,
    /// Compile to MIR and interpret it using MIRI. Default for the `debug` profile. Results in
    /// much faster compile times as it skips LLVM, but execution time is much slower.
    Mir,
}

/// Debug information to emit in the compiled binary.
///
/// A higher debug level makes it easier to debug the compiled binary if any errors occur, but
/// increases the size of the binary.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DebugInfo {
    /// No debug information. Default for the `release` profile.
    None,
    /// Emit a low amount of debug information (call stacks, item definition line/column numbers).
    /// Default for all built-in profiles except `debug` and `release`.
    Low,
    /// Emit full debug information (everything in `Low`, plus local symbols, types, spans).
    /// Default for the `debug` profile.
    Full,
}

/// Compiler configuration.
#[derive(Copy, Clone, Debug)]
pub struct CompileOptions {
    /// Compilation target.
    pub target: Target,
    /// Debug information to emit in the compiled binary. Only applicable when compiling to a
    /// native binary and not MIR.
    pub debug_info: DebugInfo,
    /// Whether to run debug assertions. Defaults to `true` for all built-in profiles except
    /// `release`.
    pub debug_assertions: bool,
}
