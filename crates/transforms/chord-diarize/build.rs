// sherpa-rs-sys links libsherpa-onnx-c-api.so dynamically without an rpath;
// bake $ORIGIN so the binary finds the sibling .so at runtime.
fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
