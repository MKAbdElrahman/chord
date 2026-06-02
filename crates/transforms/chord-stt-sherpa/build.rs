// sherpa-rs-sys ships libsherpa-onnx-c-api.so into the target dir next to our
// binary, but links it dynamically without an rpath. Bake `$ORIGIN` so the
// binary finds the sibling .so at runtime (the loader expands $ORIGIN to the
// executable's own directory). Keep the .so beside the binary when installing.
fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
