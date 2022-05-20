# Terbium
The performant yet elegant and feature-packed programming language. Implemented with Rust.

## Installation
Terbium is still a work in progress. You can contribute by simply cloning this repository, however.

## Hello, world!
```ts
require std;

func main() {
    std.println("Hello, world!");
}
```

## Goals
We want Terbium to meet the following:

- A language that doesn't take long to learn
- A language emphasizing strong and static types
  - Types are static, but type inference will exist
- A language which is fast and performant
- A language which is elegant to write and emphasizes readability
- A language (compiler/interpreter) that catches bugs before runtime

### Static types?
We want to enforce a static type system in Terbium that isn't too restrictive:

- Optionally allow a variable to keep a constant type throughout its lifetime (`let`)
- Default all types to the `auto` type (Type inference)
  - When a type cannot be inferred, use the `any` type unless explicitly disabled (`@trb:strict`)
- Allow a robust type system (e.g. Generics)

Terbium designs static types like this so that while beginners don't have to learn Terbium with
the complexity of a type system, and gradually implement these types as they learn more about them.

#### Ran at compile-time
The static-type checker, along with other types of analysis (such as optimization),
is run at compile-time\*. This ensures that your code will run as expected with minimal errors
before even running.

\* If you run Terbium using its interpreter, analysis will be run immediately before runtime.

#### Yes, we have `null`.
A very common pattern in more modernized languages are those which lack a
"null" type.

We chose to keep `null` but make it required to explicitly handle and propagate `null`,
specifically with the static type-checker which is run at compile-time.
Luckily, doing such is [easy](#efficient-null-handling).

### Concise Code
We also want to make Terbium code as concise as possible, without sacrificing performance or readability.

#### Examples:

##### Statements are Expressions
Directly assign the result of an `if` statement to a variable:
```ts
let x = if 1 + 1 == 2 {
    std.println("1 + 1 is indeed 2!");
    2
}

std.println(x); // 2
```

Call a function with the result of a `match` statement:
```ts
std.println(match 1 + 1 {
   1 -> "one",
   2 -> "two",
   else "other",       
});
```

or... use non-expressions with `match`:
```ts
std.println(match 1 + 1 {
    2 -> {
       let message = "two";
       message
    },
    else {
        let message = "other";
        message
    },
});
```

(above three inspired by Rust)

##### Efficient `null` handling:

Null propagations and assertions:
```ts
std.println(x.y?.z!);

// equivalent to:
// if x.y { if x.y.z { std.println(x.y.z) } else { throw error } } else { std.println("null") }
```

(above inspired by TypeScript)

## Repository Navigation
- [terbium_analyzer](https://github.com/TerbiumLang/Terbium/tree/main/terbium_analyzer):
  Analyzes Terbium code and houses the static type-checker. This is run at compile-time
  and apart from ensuring type-safety, will try picking as many places that could be
  optimized, and as many places that could lead to errors at runtime.
- [terbium_grammar](https://github.com/TerbiumLang/Terbium/tree/main/terbium_grammar):
  Tokenizes Terbium code and transforms it into an
  [Abstract Syntax Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) (AST).
- [terbium_interpreter](https://github.com/TerbiumLang/Terbium/tree/main/terbium_interpreter):
  The interpreter which interprets and evaluates code generated from ``terbium_grammar``.

## Contributing
See [CONTRIBUTING.md](https://github.com/TerbiumLang/Terbium/blob/main/CONTRIBUTING.md) for details.

## Credits
#### Terbium is a project by **[jay3332](https://github.com/jay3332)**.

Thanks to the following core contributors:
- [Cryptex-github](https://github.com/Cryptex-github): Creating workflows + General contributions
