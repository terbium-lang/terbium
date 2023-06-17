# Terbium

Write bug-free code that runs at native performance with ease by using Terbium, a high-level programming language that doesn't compromise in performance, readability, reliability and developer experience. 
This is the rewrite branch of Terbium. See legacy Terbium [here](https://github.com/TerbiumLang/Terbium/tree/legacy).

## Specification

A semi-formal language specification can be found [here](https://jay3332.gitbook.io/terbium/spec). The specification may be outdated with this specific implementation.

## Installation

Terbium is still a work in progress. If you want to:

* *contribute to Terbium*, you can clone this repository. See [CONTRIBUTING.md](https://github.com/terbium-lang/terbium/blob/main/CONTRIBUTING.md) for more information.
* *use Terbium*, see [INSTALLATION.md](https://github.com/terbium-lang/terbium/blob/main/INSTALLATION.md) for more information.

## Why Terbium?

Terbium is designed to be:

* **Concise**. Write less code wherever possible. High-level constructs and language features make Terbium elegant to write.
* **Readable**. Being explicit and unambiguous in the code where needed while still being concise.
* **Simple**. Complexity is abstracted away from the developer when writing a Terbium program, but low-level features are still accessable whenever needed.
* **Performant**. With LLVM, Terbium can compile to optimized native machine code, so your high-level code won't compromise runtime performance.
* **Robust**. Write error-free code with Terbium. Null is a concept that does not exist in Terbium, and all errors must explicitly be handled. Mutability is explicit and memory is automatically freed.
* **Rapid**. Rapid-prototyping and developer experience is just as important as runtime performance. A *debug mode* compiles semi-optimized code that compiles fast and suppresses pedantic errors as warnings.
* **Typed**. A comprehensive type system allows Terbium to infer the types of most values and functions at compile-time. If a type cannot be inferred, it must be explicitly specified.

## Examples

### Hello, world!

```swift
// Main function is recommended but not required
func main() {
    println("Hello, world!");
}
```

### Fibonacci

```swift
func fib(n: uint) = if n <= 1 then n else fib(n - 1) + fib(n - 2);

func main() {
    (0..=10).map(fib).for_each(println);
}
```

### Optional values

```swift
// Optional values are of type T? and can be either .some(value) or .none
// Any `value` of type T can coerce into T? by .some(value).

func maybe_five() -> int? = .none;

func main() {
    println(maybe_five() else 10); // 10
}
```
### Error handling

```swift
import std.io.{Error, stdout};

// T throws E is a type itself:
type Result<T> = T throws Error;

// throws E expands to void throws E, void being the unit type
type VoidResult = throws Error;

// Specify the return type as `throws Error` to indicate that the return value might fail
func fallible() -> throws Error {
    let mut out = stdout();
    out.write("This might fail")!; // use ! to propagate the error to the function

    // implied return value of `void` is coerced to `.ok(void)`
    // in fact, any `value` of type T can be coerced to `T throws _` as `.ok(value)`
    //
    // this explicit type cast will result in `.ok(5)`:
    // let x = 5 to /* `to` is the cast operator */ int throws Error;
}

func main() {
    // Use `catch` to handle a fallible value
    fallible() catch err {
        eprintln($"Failed to print: {err}"); // string interpolation
    };

    // If there is absolutely no way to handle an error,
    // you can halt the program's execution with `panic`:
    //
    // Note: `else` is equivalent to `catch _`
    fallible() else panic("Oh no!");
    // You can also use `.unwrap()` to panic with no message:
    fallible().unwrap();
}
```
