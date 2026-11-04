{
  description = "Development environment for rust-qldpc";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python312.withPackages (ps: with ps; [ numpy scipy ]);
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              maturin
              pkg-config
              rustc
              rustfmt
              uv
              python
            ];

            PYO3_PYTHON = "${python}/bin/python3";

            shellHook = ''
              export LD_LIBRARY_PATH="$(${python}/bin/python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };
        });
    };
}
