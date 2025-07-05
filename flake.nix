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

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };
      in {
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

            libsixel

            linuxPackages_latest.perf

            # WINIT_UNIX_BACKEND=wayland
            wayland

            # WINIT_UNIX_BACKEND=x11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            xorg.libX11
          ];

          LD_LIBRARY_PATH = inputs.nixpkgs.lib.makeLibraryPath buildInputs;
          XDG_DATA_DIRS =
            "${pkgs.gtk3}/share/gsettings-schemas/gtk+3-${pkgs.gtk3.version}:"
            + "${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/gsettings-desktop-schemas-${pkgs.gsettings-desktop-schemas.version}";
        };
      });
}
