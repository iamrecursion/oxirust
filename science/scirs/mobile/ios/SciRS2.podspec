Pod::Spec.new do |s|
  s.name             = 'SciRS2'
  s.version          = '0.2.0'
  s.summary          = 'Scientific Computing Library for iOS - Pure Rust Implementation'
  s.description      = <<-DESC
    SciRS2 brings high-performance scientific computing to iOS with:
    - ARM NEON-optimized SIMD operations
    - Metal GPU acceleration
    - Battery-efficient algorithms
    - Thermal-aware processing
    - Neural network operations
    - Linear algebra (BLAS/LAPACK)
    - Signal processing and FFT
    Pure Rust implementation with no C/Fortran dependencies.
  DESC

  s.homepage         = 'https://github.com/cool-japan/scirs'
  s.license          = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.author           = { 'COOLJAPAN OU' => 'team@kitasan.dev' }
  s.source           = { :git => 'https://github.com/cool-japan/scirs.git', :tag => s.version.to_s }

  s.ios.deployment_target = '13.0'
  s.swift_version = '5.9'

  # Swift wrapper
  s.source_files = 'mobile/ios/*.swift'

  # Native library (pre-built XCFramework)
  s.vendored_frameworks = 'target/ios/SciRS2.xcframework'

  # Framework dependencies
  s.frameworks = 'Metal', 'Accelerate'

  # Build settings
  s.pod_target_xcconfig = {
    'ENABLE_BITCODE' => 'YES',
    'OTHER_LDFLAGS' => '-lc++ -lm'
  }

  # Requires iOS 13+ for Metal Performance Shaders
  s.requires_arc = true
end
