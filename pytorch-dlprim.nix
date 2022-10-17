{ stdenv, fetchFromGitHub, cmake, python3Packages, ocl-icd, opencl-clhpp
, dlprimitives }:

stdenv.mkDerivation {
  pname = "pytorch-dlprim";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "pytorch_dlprim";
    sha256 = "sha256-cWjjqvmFbYs4a8Vkx255zjuHnAK5qJvcjhgHQZc9yQo=";
    rev = "58de5cfd922f0d7d11f5b5da094b27c82d095519";
  };

  installPhase = ''
    mkdir -p $out/lib
    cp libpt_ocl.so $out/lib
  '';

  nativeBuildInputs = [ cmake ];
  buildInputs = [ ocl-icd opencl-clhpp python3Packages.torch dlprimitives ];
}
