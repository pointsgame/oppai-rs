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
        devShell = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            (fenix.combine [
              (fenix.latest.withComponents [
                "cargo"
                "clippy"
                "rust-src"
                "rustc"
                "rustfmt"
              ])
              fenix.targets.wasm32-unknown-unknown.latest.rust-std
            ])
            rust-analyzer
            wasm-bindgen-cli
            trunk
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

            xorg.libX11
            xorg.libXrandr
            xorg.libXcursor
            xorg.libXi
            libxkbcommon
            libGL
            fontconfig
            wayland
            # (python3.withPackages (pkgs: with pkgs; [ pytorch torchvision ]))
            # (pkgs.callPackage ./pytorch-dlprim.nix { })
          ];

          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
          XDG_DATA_DIRS =
            "${pkgs.gtk3}/share/gsettings-schemas/gtk+3-${pkgs.gtk3.version}:"
            + "${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/gsettings-desktop-schemas-${pkgs.gsettings-desktop-schemas.version}";
        };
      });
}
