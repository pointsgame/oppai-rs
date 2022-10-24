{ stdenv, fetchFromGitHub, cmake, python3Packages, ocl-icd, opencl-clhpp
, dlprimitives }:

stdenv.mkDerivation {
  pname = "pytorch-dlprim";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "pytorch_dlprim";
    sha256 = "sha256-et7psx+WP7KR01Op4cMJXdDNa9xY6zpYccJgedpRohg=";
    rev = "525956ab8fd549210aaafdc0908ee31afaf99211";
  };

  installPhase = ''
    mkdir -p $out/lib
    cp libpt_ocl.so $out/lib
  '';

  nativeBuildInputs = [ cmake ];
  buildInputs = [ ocl-icd opencl-clhpp python3Packages.torch dlprimitives ];
}
