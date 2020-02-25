# exact diagonalization

* Python and Julia codes of exact diagonalization

* Use sparse matrices

* Sz conserved / Sz nonconserved

* Sz conserved case: algorithm based on
  [titpack2](http://www.qa.iir.titech.ac.jp/~nishimori/titpack2_new/index-e.html)
  and
  [HPhi](https://github.com/issp-center-dev/HPhi)

  * Use the snoob function for finding the next higher number after a given number that has the same number of 1-bits (down spins)

    * See
      [Hacker's Delight](https://www.hackersdelight.org/hdcodetxt/snoob.c.txt)
      [Hacker's Delight (web archive)](http://web.archive.org/web/20190109000030/https://www.hackersdelight.org/hdcodetxt/snoob.c.txt)

    * See also [HPhi tips](http://www.pasums.issp.u-tokyo.ac.jp/wp-content/themes/HPhi/media/develop/tips.pdf)
      [HPhi tips (mirror)](http://issp-center-dev.github.io/HPhi/develop/tips.pdf)

    * For related bitwise operations, see
      [Count bits](http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel)
      [Anothr count bits](https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer)

  * Use two-dimensional search

    * See [DOI:10.1103/PhysRevB.42.6561](https://doi.org/10.1103/PhysRevB.42.6561)

    * See also
      [manual of titpack2 (en)](http://www.qa.iir.titech.ac.jp/~nishimori/titpack2_new/index-e.html)
      [manual of titpack2 (jp)](http://hdl.handle.net/2433/94584)

  * Without generating matrix elements to save memory (test version, not so efficient...)

    * Use scipy.sparse.linalg.LinearOperator for Python

    * Use [LinearMap](https://github.com/Jutho/LinearMaps.jl) for Julia

      * Tutorials in Japanese
        https://qiita.com/cometscome_phys/items/9f9cbd4f2ffa75e09f5e
        https://qiita.com/cometscome_phys/items/643f56d365c095884b89

## Models

* 1D J1-J2 Heisenberg model

  * See, for example, numerical results in
    [DOI:10.1016/0375-9601(92)90823-5](https://doi.org/10.1016/0375-9601(92)90823-5)
    [DOI:10.1103/PhysRevB.54.R9612](https://doi.org/10.1103/PhysRevB.54.R9612)

  * See also analytical results in
    [DOI:10.1063/1.1664978](https://doi.org/10.1063/1.1664978)
    [DOI:10.1063/1.1664979](https://doi.org/10.1063/1.1664979)
    [DOI:10.1103/PhysRevB.25.4925](https://doi.org/10.1103/PhysRevB.25.4925)

* 2D J1-J2 Heisenberg model on a square lattice

  * See, for example, numerical results in
    [DOI:10.1103/PhysRevLett.63.2148](https://doi.org/10.1103/PhysRevLett.63.2148)
    [DOI:10.1051/jp1:1996236](https://doi.org/10.1051/jp1:1996236)

  * See also [DOI:10.1016/j.cpc.2018.08.014](https://doi.org/10.1016/j.cpc.2018.08.014)
    (spin correlations at J2=0)

## Useful references

* Papers 

  * [HPhi DOI:10.1016/j.cpc.2017.04.006](https://doi.org/10.1016/j.cpc.2017.04.006)
    [(arXiv)](https://arxiv.org/abs/1703.03637)

  * [Sz and momentum conservation, Lanczos DOI:10.1063/1.3518900](https://doi.org/10.1063/1.3518900)
    [(arXiv)](https://arxiv.org/abs/1101.3281)

  * [2D search of states DOI:10.1103/PhysRevB.42.6561](https://doi.org/10.1103/PhysRevB.42.6561)

  * [search by Pascal's triangle arXiv:1912.11240](https://arxiv.org/abs/1912.11240)

* Codes / Lecture notes

  * Anders W. Sandvik
    http://physics.bu.edu/~sandvik/vietri/index.html
    http://physics.bu.edu/~sandvik/vietri/vietri.pdf
    http://physics.bu.edu/~sandvik/vietri/dia.pdf

    * Use a bisectional search instead of making a 2D search list
    * Implementation may be easier with momentum conservation

  * Frank Pollmann
    http://tccm.pks.mpg.de/?page_id=871
    https://www.pks.mpg.de/~frankp/comp-phys/
    https://www.pks.mpg.de/~frankp/comp-phys/exact_diagonalization_conserve.py

  * Guillaume Roux
    http://lptms.u-psud.fr/wiki-cours/index.php/Lectures_on_Exact_Diagonalization
    http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture1.html
    http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html

  * Alexander Wietek
    https://github.com/alexwie/ed_basics

    * State search by a python built-in function index() for list (np.where for numpy array)

  * Glen Evenbly
    https://www.tensors.net/exact-diagonalization

  * Ryan Levy
    https://ryanlevy.github.io/physics/Heisenberg1D-ED/
