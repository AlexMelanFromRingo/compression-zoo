use std::io::Read;
fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    // Try increasing prefix lengths to find first panic
    let mut lo = 1usize;
    let mut hi = buf.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let result = std::panic::catch_unwind(|| {
            let payload = sevenz::ppmd8::encode_one_shot(
                &buf[..mid], 1u32 << 20, 6,
                sevenz::ppmd8::RestoreMethod::Restart);
            payload.len()
        });
        match result {
            Ok(_) => lo = mid + 1,
            Err(_) => hi = mid,
        }
    }
    eprintln!("First panic at prefix length: {}", hi);
}
