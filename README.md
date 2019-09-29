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
  * 2D J1-J2 Heisenberg model on a square lattice
