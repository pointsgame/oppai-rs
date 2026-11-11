{
  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      ref = "nixos-unstable";
    };

    flake-utils = {
      type = "github";
      owner = "numtide";
      repo = "flake-utils";
    };

    fenix = {
      type = "github";
      owner = "nix-community";
      repo = "fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };

        # The LLVM the ROCm backend compiles its kernels with. Its build script
        # downloads a prebuilt bundle whose llvm-config cannot run on NixOS, so
        # the same bundle is provided patched instead, at the place the build
        # script looks for it before downloading. The version has to follow
        # tracel-llvm-bundler in Cargo.lock.
        tracel-llvm = pkgs.stdenv.mkDerivation rec {
          pname = "tracel-llvm";
          version = "23.1.0-3";
          src = pkgs.fetchurl {
            url = "https://github.com/tracel-ai/tracel-llvm/releases/download/v${version}/linux-x64.tar.xz";
            sha256 = "1a2b381da394078c82646bd118eb7dceb56320b700a7ef879468e6175a315c1a";
          };
          nativeBuildInputs = [ pkgs.autoPatchelfHook ];
          buildInputs = [ pkgs.stdenv.cc.cc.lib ];
          dontConfigure = true;
          dontBuild = true;
          dontStrip = true;
          # The bundle links ld64.lld to an lld it does not ship; nothing here
          # uses either.
          dontCheckForBrokenSymlinks = true;
          installPhase = ''
            cp -r . $out
            # Marks the bundle as installed, so that the build script uses it
            # as it is.
            touch $out/.tracel-llvm-installed
          '';
        };
      in
      {
        devShell = pkgs.mkShell rec {
          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            (fenix.combine [
              (fenix.stable.withComponents [
                "cargo"
                "clippy"
                "rust-src"
                "rustc"
                "rustfmt"
              ])
              fenix.targets.wasm32-unknown-unknown.stable.rust-std
              fenix.targets.wasm32-wasip1.stable.rust-std
            ])
            rust-analyzer
            wasm-bindgen-cli
            trunk
            wasmtime
            wasm-pack
          ];

          buildInputs = with pkgs; [
            atk
            cairo
            freetype
            gdk-pixbuf
            glib
            gtk3
            librsvg
            pango
            vulkan-loader

            libxkbcommon
            libGL

            perf

            # ROCm
            rocmPackages.clr
            rocmPackages.hipblas
            rocmPackages.rocblas

            # for rfd
            dbus

            # WINIT_UNIX_BACKEND=wayland
            wayland

            # WINIT_UNIX_BACKEND=x11
            libXcursor
            libXrandr
            libXi
            libX11
          ];

          # The ROCm backend links LLVM, which is C++ and needs libstdc++ at run
          # time.
          LD_LIBRARY_PATH = inputs.nixpkgs.lib.makeLibraryPath (buildInputs ++ [ pkgs.stdenv.cc.cc.lib ]);

          shellHook = pkgs.lib.optionalString (system == "x86_64-linux") ''
            tracel_llvm="''${XDG_DATA_HOME:-$HOME/.local/share}/tracel/tracel-llvm-${tracel-llvm.version}"
            if [ -L "$tracel_llvm" ] || [ ! -e "$tracel_llvm" ]; then
              mkdir -p "$(dirname "$tracel_llvm")"
              ln -sfn ${tracel-llvm} "$tracel_llvm"
            elif ! "$tracel_llvm/bin/llvm-config" --version >/dev/null 2>&1; then
              echo "warning: $tracel_llvm holds an LLVM that cannot run here, remove it to use the patched one" >&2
            fi
          '';
          XDG_DATA_DIRS =
            "${pkgs.gtk3}/share/gsettings-schemas/gtk+3-${pkgs.gtk3.version}:"
            + "${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/gsettings-desktop-schemas-${pkgs.gsettings-desktop-schemas.version}";
        };
      }
    );
}
