from pyscf.lo import PM
from pyscf.lib import logger

import numpy
import scipy

import time

import common


def get_lmp2_residuals(mp2_t2, oovv, fock_occ, fock_virt, ovlp, sparsity_desc, kind="ft-first"):
    """
    Calculates LMP2 residuals.
    Args:
        mp2_t2 (dict): sparse LMP2 amplitudes;
        oovv (dict): sparse electron repulsion integrals;
        fock_occ (numpy.ndarray): occupied Fock matrix;
        fock_virt (numpy.ndarray): virtual Fock matrix;
        ovlp (numpy.ndarray): overlap matrix in the virtual space;
        sparsity_desc (dict): lists of PAO basis functions for each pair;
        kind (str): specifies the order of the tensor product;

    Returns:
        Sparse residuals for the given amplitudes.
    """

    def oovv_stf_fts(mp2_t2, oovv, fock_virt, ovlp, sparsity_desc):
        result = {}
        for k in sparsity_desc:
            # T2 indexes
            ti = sparsity_desc[k]
            # Sparsify S, F
            s = ovlp[ti, :][:, ti]
            f = fock_virt[ti, :][:, ti]
            result[k] = numpy.einsum("ab,ar,bs->rs", mp2_t2[k], f, s)
            result[k] += numpy.einsum("ab,ar,bs->rs", mp2_t2[k], s, f)
            result[k] += oovv[k]
        return result

    def overlap_sandwich(mp2_t2, ovlp, sparsity_desc):
        result = {}
        for k in sparsity_desc:
            # T2 indexes
            ti = sparsity_desc[k]
            # Sparsify S, F
            s = ovlp[ti, :][:, ti]
            result[k] = numpy.einsum("ab,ar,bs->rs", mp2_t2[k], s, s)
        return result

    def product_with_lmo_fock(sts, fock_occ, sparsity_desc):
        nocc = fock_occ.shape[0]
        result = dict((k, numpy.zeros(v.shape)) for k, v in sts.items())
        for k in sparsity_desc:
            for i in range(nocc):
                for k_source, k_fock in (
                        ((k[0], i), (i, k[1])),
                        ((i, k[1]), (k[0], i)),
                ):
                    index_avail = sparsity_desc[k_source]
                    index_req = sparsity_desc[k]
                    index_ovlp = numpy.intersect1d(index_avail, index_req)
                    source = numpy.where(index_avail[:, numpy.newaxis] == index_ovlp[numpy.newaxis, :])[0]
                    source = numpy.ix_(source, source)
                    dest = numpy.where(index_req[:, numpy.newaxis] == index_ovlp[numpy.newaxis, :])[0]
                    dest = numpy.ix_(dest, dest)
                    result[k][dest] += sts[k_source][source] * fock_occ[k_fock]
        return result

    result = oovv_stf_fts(mp2_t2, oovv, fock_virt, ovlp, sparsity_desc)
    if kind == "sts-first":
        sts = overlap_sandwich(mp2_t2, ovlp, sparsity_desc)
        sfts = product_with_lmo_fock(sts, fock_occ, sparsity_desc)
    elif kind == "ft-first" or kind == "tf-first":
        ft = product_with_lmo_fock(mp2_t2, fock_occ, sparsity_desc)
        sfts = overlap_sandwich(ft, ovlp, sparsity_desc)
    else:
        raise ValueError("Unknown kind: {}".format(kind))

    for k in sparsity_desc:
        result[k] -= sfts[k]
    return result


def get_lmp2_correction(r_pao, fock_occ, fock_basis_local, fock_energies_local):
    """
    Transforms residuals into correction to the LMP2 amplitudes.
    Args:
        r_pao (dict): sparse residuals;
        fock_occ (numpy.ndarray): occupied Fock matrix;
        fock_basis_local (dict): local virtual (PAO) basis for the pairs;
        fock_energies_local (dict): local virtual (PAO) basis eigenvalues for the pairs;

    Returns:
        A sparse correction to the LMP2 amplitudes.
    """
    result = {}
    t2_diff = 0

    for k, v in r_pao.items():
        basis = fock_basis_local[k]
        virt_e = fock_energies_local[k]
        # dual = numpy.linalg.inv(basis).T
        v = common.transform(v, basis)

        denominator = fock_occ[k[0], k[0]] + fock_occ[k[1], k[1]] - virt_e[:, numpy.newaxis] - virt_e[numpy.newaxis, :]
        dt = v / denominator
        dt_pao = common.transform(dt, basis.T)

        result[k] = dt_pao
        t2_diff = max(t2_diff, numpy.abs(dt_pao).max())

    return result, t2_diff


def get_lmp2_energy(mp2_t2, oovv):
    """
    Calculates the LMP2 energy.
    Args:
        mp2_t2 (dict): sparse LMP2 amplitudes;
        oovv (dict): sparse electron repulsion integrals;

    Returns:
        The LMP2 energy value.
    """
    result = 0
    for k, t2 in mp2_t2.items():
        t2_conj = mp2_t2[k[::-1]]
        result += numpy.sum(oovv[k] * (2 * t2 - t2_conj))
    return result


def get_ao_ownership_matrix(mol):
    """
    Retrieves a matrix defining which atoms atomic orbitals belong to.
    Args:
        mol (pyscf.Mole): a Mole object;

    Returns:
        A 2D [basis size x number of atoms] numpy array of bool values where truth values indicate the fact that
        a particular atomic orbital belongs to a particular atom.
    """
    ao_labels = mol.ao_labels(fmt=False)
    result = numpy.zeros([len(ao_labels), mol.natm], dtype=bool)
    for i, j in enumerate(ao_labels):
        result[i, j[0]] = True
    return result


def get_mulliken(mo, ovlp):
    """
    Mulliken charge analysis for molecular orbitals.
    Args:
        mo (numpy.ndarray): molecular orbitals;
        ovlp (numpy.ndarray): the overlap matrix;

    Returns:
        An array with partial weights of molecular orbitals on atomic orbitals.
    """
    a = numpy.einsum("ai,bi,ab->ai", mo, mo, ovlp)
    b = numpy.einsum("ai,bi,ab->bi", mo, mo, ovlp)
    return 0.5 * (a + b)


def get_dominating_contributions(weight_lmo_atom, criterion=0.98, log=None):
    """
    Retrieves lists of atoms contributing the largest weights to each of molecular orbitals.
    Args:
        weight_lmo_atom (numpy.ndarray): an array with partial weights of molecular orbitals on atoms;
        criterion (float): a criterion to terminate the summation of weights;
        log (any pyscf logging object): log to output additional info to;

    Returns:
        A list where each item is a list with atomic indexes specifying atoms where a particular molecular orbital
        resides.
    """
    total_lmo_charge = weight_lmo_atom.sum(axis=1)
    result = []
    for i, tot in zip(weight_lmo_atom, total_lmo_charge):
        s = numpy.argsort(i)[::-1]
        w = numpy.cumsum(i[s])
        last = (w < tot * criterion).sum()
        result.append(s[:last + 1].tolist())
    if log is not None:
        logger.info(log, "Localized orbitals:")
        for i, (ats, tot) in enumerate(zip(result, total_lmo_charge)):
            logger.info(log, "  state {:d} carries charge {:.3f}, localized @ {:d} cites {} sum {:.3f} > {:.3f}".format(
                i,
                tot,
                len(ats),
                ", ".join(list(
                    "#{:d}({:.3f})".format(j, weight_lmo_atom[i, j]) for j in ats
                )),
                weight_lmo_atom[i][ats].sum(),
                criterion * tot,
            ))
    return result

def iter_local_conventional(mol, mo_occ):
    """
    Performs iterations over conventional pair subspaces of local MP2.
    Args:
        mol (pyscf.Mole): a Mole object;
        mo_occ (numpy.ndarray): occupied molecular orbitals;

    Returns:
        For each pair, returns molecular orbital indexes i, j, a set of atomic indexes corresponding to this pair and
        a numpy array with atomic orbital indexes corresponding to this pair.
    """
    ao_ownership = get_ao_ownership_matrix(mol)
    mulliken_ao_lmo = get_mulliken(mo_occ, mol.intor_symmetric('int1e_ovlp'))
    mulliken_lmo_atom = numpy.dot(mulliken_ao_lmo.T, ao_ownership)
    lmo_ownership = get_dominating_contributions(mulliken_lmo_atom, log=mol)

    for i in range(mo_occ.shape[1]):
        a1 = lmo_ownership[i]
        for j in range(mo_occ.shape[1]):
            a2 = lmo_ownership[j]
            # Debug print
            print(f"Pair ({i},{j}): a1 = {a1}, a2 = {a2}")            
            # For diagonal pairs, only use the localization atoms of that orbital
            if i == j:
                local_atoms = sorted(set(a1))
            else:
                local_atoms = sorted(set(a1) | set(a2))
            
            orbs = ao_ownership[:, local_atoms].sum(axis=1)
            yield i, j, local_atoms, numpy.argwhere(orbs)[:, 0]

class AbstractLMP2IntegralProvider(common.SimpleCachingIntegralProvider):

    def get_eri_diagonal_block(self, atoms):
        """
        Retrieves a subset of electron repulsion integrals corresponding to a given subset of atomic basis functions.
        Args:
            atoms (list, tuple): a subset of atoms where the basis functions reside;

        Returns:
            A four-index tensor with ERIs belonging to a given subset of atoms.
        """
        raise NotImplementedError()

    def get_lmo_pao_block(self, atoms, orbitals, lmo1, lmo2, pao):
        """
        Retrieves a block of electron repulsion integrals in the localized molecular orbitals / projected atomic orbitals basis
        set.
        Args:
            atoms (list): a list of atoms within this space;
            orbitals (list): a list of orbitals within this space;
            lmo1 (numpy.ndarray): localized molecular orbital 1;
            lmo2 (numpy.ndarray): localized molecular orbital 2;
            pao (numpy.ndarray): projected atomic orbitals;

        Returns:
            A block of electron repulsion integrals.
        """
        raise NotImplementedError()


class SimpleLMP2IntegralProvider(AbstractLMP2IntegralProvider):

    def get_eri_diagonal_block(self, atoms):
        """
        See parent description.
        """
        return self.get_eri(atoms, atoms, atoms, atoms)

    get_eri_diagonal_block.__doc__ = AbstractLMP2IntegralProvider.get_eri_diagonal_block.__doc__

    def get_lmo_pao_block(self, atoms, orbitals, lmo1, lmo2, pao):
        """
        See parent description.
        """

        oovv = self.get_eri_diagonal_block(atoms).swapaxes(1, 2)
        lmo1 = lmo1[orbitals]
        lmo2 = lmo2[orbitals]
        pao = pao[numpy.ix_(orbitals, orbitals)]
        result = common.transform(
            common.transform(
                common.transform(oovv, lmo1[:, numpy.newaxis], axes=0),
                lmo2[:, numpy.newaxis],
                axes=1,
            ),
            pao,
            axes='l2',
        )
        return result[0, 0]

    get_lmo_pao_block.__doc__ = AbstractLMP2IntegralProvider.get_lmo_pao_block.__doc__


class LMP2(object):
    def __init__(
            self,
            mf,
            localization_provider=PM,
            local_space_provider=iter_local_conventional,
            local_integral_provider=SimpleLMP2IntegralProvider,
    ):
        """
        A local MP2 implementation by Pulay and Saebo. The sparse MP2 integrals and amplitudes are stored as dicts where
        keys are index pairs of local molecular orbitals and values are square matrices in projected atomic orbital
        basis sets.
        Args:
            mf (pyscf.scf.*): a mean-field solution to the given system;
            localization_provider (pyscf.lo.*): pyscf localization provider;
            local_space_provider (iterable): an iterable yielding two pair indices, a list of indexes of atoms
            corresponding to a given subspace and a list of atomic orbitals corresponding to a given subspace;

            local_integral_provider (class): a class implementing calculation of blocks of four-center integrals;
        """
        self.mf = mf
        self.localization_provider = localization_provider
        self.local_space_provider = local_space_provider
        self.local_integral_provider = local_integral_provider

        self.initialized_local_integral_provider = None

        # Dense
        self.fock_lmo = None
        self.fock_pao = None
        self.ovlp_pao = None

        # Sparse
        self.domain_orbital_map = None
        self.domain_atom_map = None
        self.domain_eri = None
        self.domain_fock_basis = None
        self.domain_fock_energies = None
        self.t2 = None

        # Energy
        self.emp2 = None

        self.convergence_history = []

    def get_mol(self):
        """
        Retrieves the Mole object.
        Returns:
            The Mole object.
        """
        return self.mf.mol

    def get_mo_occupied(self):
        """
        Retrieves occupied orbitals.
        Returns:
            A rectangular matrix with occupied orbitals.
        """
        return self.mf.mo_coeff[:, :self.get_mol().nelectron // 2]

    def get_mo_virtual(self):
        """
        Retrieves virtual orbitals.
        Returns:
            A rectangular matrix with virtual orbitals.
        """
        return self.mf.mo_coeff[:, self.get_mol().nelectron // 2:]

    def get_mo_localized(self):
        """
        Calculates localized occupied molecular orbitals.
        Returns:
            A rectangular matrix with localized occupied molecular orbitals.
        """
        if self.localization_provider is None:
            return self.get_mo_occupied()
        else:
            return self.localization_provider(self.get_mol()).kernel(
                self.get_mo_occupied(),
                verbose=self.mf.verbose,
            )

    def get_pao_projection_matrix(self):
        """
        Calculates the projection matrix onto the virtual space.
        Returns:
            The matrix in AO basis projecting onto the virtual space.
        """
        mo_virtual = self.get_mo_virtual()
        virtual_density_matrix = numpy.dot(mo_virtual, mo_virtual.T)
        return numpy.dot(
            virtual_density_matrix,
            self.mf.get_ovlp(),
        )

    def build(self):
        """
        Updates key parameters of LMP2 iterations.
        """

        # Localization
        mo_loc = self.get_mo_localized()

        # PAO projection
        projection_matrix = self.get_pao_projection_matrix()

        # Fock and overlap matrices
        #fock = self.mf.get_fock()
        dual = numpy.linalg.inv(self.mf.mo_coeff).T
        fock = numpy.einsum("ij,j,kj->ik", dual, self.mf.mo_energy, dual)
        ovlp = self.mf.get_ovlp()

        # Transformed Fock and overlap matrices
        self.fock_lmo = common.transform(fock, mo_loc)
        self.fock_pao = common.transform(fock, projection_matrix)
        self.ovlp_pao = common.transform(ovlp, projection_matrix)

        self.initialized_local_integral_provider = self.local_integral_provider(self.get_mol())
        self.domain_orbital_map = {}
        self.domain_atom_map = {}
        self.domain_eri = {}
        self.domain_fock_basis = {}
        self.domain_fock_energies = {}
        self.t2 = {}
        for i, j, atoms, orbitals in self.local_space_provider(self.get_mol(), mo_loc):
            self.domain_orbital_map[i, j] = orbitals
            self.domain_atom_map[i, j] = atoms

            self.domain_eri[i, j] = self.initialized_local_integral_provider.get_lmo_pao_block(
                atoms,
                orbitals,
                mo_loc[:, i],
                mo_loc[:, j],
                projection_matrix,
            )

            local_fock = self.fock_pao[numpy.ix_(orbitals, orbitals)]
            local_ovlp = ovlp[numpy.ix_(orbitals, orbitals)]
            energies, states = scipy.linalg.eigh(local_fock, local_ovlp)
            self.domain_fock_energies[i, j] = energies
            self.domain_fock_basis[i, j] = states
            self.t2[i, j] = numpy.zeros((len(orbitals),) * 2, dtype=self.domain_eri[i, j].dtype)

    def update_mp2_amplitudes(self):
        """
        Performs a single iteration of the LMP2 algorithm.
        Returns:
            A delta factor as a maximal absolute value of the difference between old and new amplitudes. The new
            amplitude is stored in self.t2.
        """
        r_pao = get_lmp2_residuals(
            self.t2,
            self.domain_eri,
            self.fock_lmo,
            self.fock_pao,
            self.ovlp_pao,
            self.domain_orbital_map,
        )
        mp2_t2_d, t2_diff = get_lmp2_correction(
            r_pao,
            self.fock_lmo,
            self.domain_fock_basis,
            self.domain_fock_energies,
        )
        for k in mp2_t2_d:
            self.t2[k] += mp2_t2_d[k]

        return t2_diff

    def kernel(self, tolerance=1e-4, mixer=None, raise_threshold=1e3, maxiter=100):
        """
        Performs LMP2 iterations. The energy and amplitudes are stored into self.emp2 and self.t2 respectively.
        Args:
            tolerance (float): desired tolerance of the amplitudes;
            mixer (object): optional mixer for amplitudes;
            raise_threshold (float): raises RuntimeError if the tolerance exceeds the value specified during iterations;
            maxiter (int): the maximal number of iterations;

        Returns:
            The LMP2 energy and LMP2 amplitudes.
        """
        logger.info(self.mf, "Building ...")
        self.build()
        logger.info(self.mf, "Starting LMP2 iterations ...")

        self.convergence_history = []

        while True:
            t_start = time.time()
            t2_diff = self.update_mp2_amplitudes()

            if mixer is not None:
                # Concatenate amplitudes into a single vector
                vector = []
                for k in sorted(self.t2.keys()):
                    vector.append(numpy.reshape(self.t2[k], -1))
                vector = numpy.concatenate(vector, axis=0)

                # Update
                vector = mixer.update(vector)

                # Split vector back into sparse amplitudes
                offset = 0
                for k in sorted(self.t2.keys()):
                    shape = self.t2[k].shape
                    size = self.t2[k].size
                    self.t2[k] = numpy.reshape(vector[offset:offset + size], shape)
                    offset += size

            # Update energy
            self.emp2 = get_lmp2_energy(self.t2, self.domain_eri)
            t_end = time.time()
            logger.info(self.mf, "  E = {:.10f} delta = {:.3e} time = {:.1f} s".format(
                self.emp2,
                t2_diff,
                t_end-t_start,
            ))

            self.convergence_history.append(t2_diff)

            if t2_diff > raise_threshold:
                raise RuntimeError("Local MP2 diverges")

            if t2_diff < tolerance:
                logger.note(self.mf, 'converged LMP2 energy = %.15g', self.emp2)
                return self.emp2, self.t2

            if maxiter is not None and len(self.convergence_history) >= maxiter:
                raise RuntimeError("The maximal number of iterations {:d} reached. The error {:.3e} is still above the requested tolerance of {:.3e}".format(
                    maxiter,
                    t2_diff,
                    tolerance,
                ))
