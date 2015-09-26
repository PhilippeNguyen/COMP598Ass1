(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "conference")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "cmex10")))
   (TeX-run-style-hooks
    "latex2e"
    "IEEEtran"
    "IEEEtran10"
    "amsmath"
    "algorithmicx"
    "algpseudocode"
    "url")
   (LaTeX-add-environments
    '("IEEEbiography" LaTeX-env-args ["argument"] 1))
   (LaTeX-add-bibliographies
    "Bibliography")))

