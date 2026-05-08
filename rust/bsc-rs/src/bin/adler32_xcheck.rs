fn main() {
    let mut v = vec![0u8; 6000];
    for (i, b) in v.iter_mut().enumerate() {
        *b = (i * 37 + 17) as u8;
    }
    println!("{:08x}", bsc_rs::adler32::adler32(&v));
}
