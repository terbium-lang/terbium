# Prion
The performant yet elegant and feature-packed programming language. Implemented with Rust.

## Goals
We want Prion to meet the following:

- A language that doesn't take long to learn
- A language emphasizing strong and static types
  - Types are static, but type inference will exist
- A language which is fast and performant
- A language which is elegant to write and emphasizes readability
- A language (compiler/interpreter) that catches bugs before runtime

### Static types?
We want to enforce a static type system in Prion that isn't too restrictive:

- Optionally allow a variable to keep a constant type throughout its lifetime (`let`)
- Default all types to the `auto` type (Type inference)
  - When a type cannot be inferred, use the `any` type unless explicitly disabled (`@pt:strict`)
- Allow a robust type system (e.g. Generics)

Prion designs static types like this so that while beginners don't have to learn Prion with
the complexity of a type system, and gradually implement these types as they learn more about them.

## Hello, world!
```ts
require std;

func main() {
    std.println("Hello, world!");
}
```
