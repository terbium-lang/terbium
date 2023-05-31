# Contributing
We appreciate any contributions made to Terbium! 
Please ensure that your code meets the following requirements before contributing though:

- Have experience in both Rust and Terbium
- Run both `cargo check` and `cargo fmt` if you have modified Rust code
    - (`cargo fmt` will handle this) Make sure each line is under 120 characters long
    - We also advise you to run `cargo test` as well, but that will be tested by the GitHub workflow check.
      Doing such will probably save you an unnecessary commit if the check does fail.
    - For a similar reason, it is also advised that you run clippy with the following command as well:
      `cargo clippy --workspace --all-features -- -D clippy::all -D clippy::pedantic -D clippy::nursery`
      and make fixes accordingly.

## Tools needed
- Rust/Cargo (Nightly)
- LLVM 16.0+

## How do I install LLVM?
1. Go to the [LLVM release page](https://github.com/llvm/llvm-project/releases) and find the latest release.

2. Pick the file that matches your system architecture.
    - For Unix/Linux platforms you want to pick the one that is prefixed with `clang+llvm-16.x.x`
    - For Windows pick [`LLVM-16.x.x-win32.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win32.exe)
      or [`LLVM-16.x.x-win64.exe`](https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win64.exe)
      depending on your system architecture

3. Extract LLVM *if* you are on Unix/Linux platforms. 
   On Linux platforms you would run `tar -xf clang+llvm-16.x.x-the-file-that-I-downloaded.tar.xz`, 
   to make it easier to type later on, it is recommended to rename this file to `llvm-16.0.0`.
   **On Windows it is already an executable so no extraction is needed.**

4. You can place the extraced LLVM folder/executable wherever you want, if you can't decide, on Windows you can place it
   in `C:/Program Files/`, on Unix/Linux you can place it in `~/`

5. Then set environment variable `LLVM_SYS_160_PREFIX=/path/to/llvm/folder/` when you are compiling Terbium. 
   For example if the LLVM folder is at `/home/user/llvm-16.0.0` then I could build Terbium via:
   `LLVM_SYS_160_PREFIX=/home/user/llvm-16.0.0 cargo build --release`

6. Building Terbium should work now.

Happy Coding!
