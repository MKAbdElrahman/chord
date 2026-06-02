// Bake $ORIGIN so the binary finds the sibling libsherpa-onnx-c-api.so.
fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
