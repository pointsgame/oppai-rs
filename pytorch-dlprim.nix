{ stdenv, fetchFromGitHub, cmake, python3Packages, ocl-icd, opencl-clhpp }:

stdenv.mkDerivation {
  pname = "pytorch-dlprim";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "pytorch_dlprim";
    sha256 = "sha256-y8MuWPmz/uIxHyhfzNf24Ru04xg4ypanRXJoQmBteok=";
    rev = "32da36d266c52250cf0fa54cdd30b847d568af2d";
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
