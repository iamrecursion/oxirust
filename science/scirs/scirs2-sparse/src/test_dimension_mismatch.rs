use crate::csr::CsrMatrix;
use crate::sym_csr::SymCsrMatrix;

#[test]
#[allow(dead_code)]
fn test_csr_to_sym_csr_dimensions() {
    // Create a simple symmetric CSR matrix
    let rows = vec![0, 0, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 0, 1, 0, 1, 2];
    let data = vec![2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0];
    let shape = (3, 3);
    
    // Create a CSR matrix
    let csr = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
    
    // Print raw content for debugging
    println!("CSR data: {:?}", csr.data);
    println!("CSR indices: {:?}", csr.indices);
    println!("CSR indptr: {:?}", csr.indptr);
    
    // Convert to SymCsrMatrix
    let sym_csr = SymCsrMatrix::from_csr(&csr);
    
    match sym_csr {
        Ok(sym) => {
            println!("SymCSR data: {:?}", sym.data);
            println!("SymCSR indices: {:?}", sym.indices);
            println!("SymCSR indptr: {:?}", sym.indptr);
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}
