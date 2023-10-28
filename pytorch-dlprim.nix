{ stdenv, fetchFromGitHub, cmake, python3Packages, ocl-icd, opencl-clhpp }:

stdenv.mkDerivation {
  pname = "pytorch-dlprim";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "pytorch_dlprim";
    sha256 = "sha256-F5gi+W439WPC1VOjYsT16VVhRpbE6Cmx4m0RefdK31w=";
    rev = "2df4ba1e316ad43fbc5ba947c6a4d5c69a2eced6";
    fetchSubmodules = true;
  };

  installPhase = ''
    mkdir -p $out/lib
    cp dlprimitives/libdlprim_core.so $out/lib
    patchelf --set-rpath "$out/lib:''$(patchelf --print-rpath libpt_ocl.so)" libpt_ocl.so
    cp libpt_ocl.so $out/lib
  '';

  nativeBuildInputs = [ cmake ];
  buildInputs = [ ocl-icd opencl-clhpp python3Packages.torch ];
}
