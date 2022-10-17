{ stdenv, fetchFromGitHub, cmake, python3, ocl-icd, opencl-clhpp, sqlite, openblas }:

stdenv.mkDerivation {
  pname = "dlprimitives";
  version = "1";

  src = fetchFromGitHub {
    owner = "artyom-beilis";
    repo = "dlprimitives";
    sha256 = "sha256-AMqoFVEnuafsMrL8q5N96/HGt6j6MkeH3X/WgGPCDWE=";
    rev = "9415f98d2866b7151d31fc80ba9376d0e08bcc74";
  };

  nativeBuildInputs = [ cmake python3 ];
  buildInputs = [ ocl-icd opencl-clhpp sqlite openblas ];
}
