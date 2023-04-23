{ stdenv, fetchFromGitHub, cmake, python3Packages, ocl-icd, opencl-clhpp }:

stdenv.mkDerivation {
  pname = "pytorch-dlprim";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "pytorch_dlprim";
    sha256 = "sha256-4+ThqhSCgiWrYuVdGPbNXMMDmlkzl2fP0oaZNmn/DDI=";
    rev = "8f62e208b1ae632b4972f9ca2ddc631f61f22cab";
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
