# Contributing
We appreciate your contributions to Terbium! Please ensure that you meet the following requirements before
contributing any code to Terbium, however:

- Have experience in both Rust and Terbium
- Run both `cargo check` and `cargo fmt` if you have modified Rust code
  - (`cargo check` will handle this) Make sure each line is under 120 characters long
  - We also advise you to run `cargo test` as well, but that will be tested by the GitHub workflow check.
    Doing such will probably save you an unnecessary commit if the check does fail.

# Tools needed
- Rust/Cargo (Nightly)
- LLVM-14.0

# How to install LLVM?
1. First go to [LLVM-14 release page](https://github.com/llvm/llvm-project/releases/tag/llvmorg-14.0.0)

2. Pick the file that matches your system architecture.
   - For *nix platforms you want to pick the one that is prefixed with `clang+llvm-14.0.0`
   - For windows pick [`LLVM-14.0.0-win32.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/LLVM-14.0.0-win32.exe) 
     or [`LLVM-14.0.0-win64.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/LLVM-14.0.0-win64.exe)
     depending on your system architecture

3. Extract LLVM if you are on *nix platforms. On linux platforms you would do `tar -xf clang+llvm-14.0.0-the-file-that-I-downloaded.tar.xz`, to make it easier to type later on, it is recommended to rename it to llvm-14.0.0. On windows is already an executable so no extraction is needed.

4. You can place the extraced LLVM folder/executable wherever you want, if you can't decide, on Windows you can place it in `C:/Program Files/`, on *nix you can place it in `./`

5. Then set env var `LLVM_SYS_140_PREFIX=/path/to/llvm/folder/` when you are compiling Terbium. For example if my llvm folder is `/home/cryptex/llvm-14.0.0` then I would compile Terbium with the following command `LLVM_SYS_140_PREFIX=/home/cryptex/llvm-14.0.0 cargo build`

6. Building Terbium should work now.


Happy Coding
