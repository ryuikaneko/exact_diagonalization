# exact diagonalization

* Python and Julia codes of exact diagonalization
* Use sparse matrices
* Sz conserved / Sz nonconserved
* Sz conserved case: algorithm based on titpack2 and [HPhi](https://github.com/issp-center-dev/HPhi)
  * Use the snoob function for finding the next higher number after a given number that has the same number of 1-bits (down spins)
    * See [Hacker's Delight](https://www.hackersdelight.org/hdcodetxt/snoob.c.txt)
    * See also [HPhi tips](http://www.pasums.issp.u-tokyo.ac.jp/wp-content/themes/HPhi/media/develop/tips.pdf)
      [HPhi tips (mirror)](http://issp-center-dev.github.io/HPhi/develop/tips.pdf)
  * Use two-dimensional search
    * See [DOI:10.1103/PhysRevB.42.6561](https://doi.org/10.1103/PhysRevB.42.6561)
    * See also [manual of titpack2](http://hdl.handle.net/2433/94584)
* Models
  * 1D Heisenberg model
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
