#pragma once

#include "../bude.h"
#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <string>

using namespace cl;

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS SyclBude

template <size_t N> class bude_kernel_ndrange;

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static constexpr sycl::access::mode RW = sycl::access::mode::read_write;
  static constexpr sycl::access::target Local = sycl::access::target::local;

  static void fasten_main(sycl::handler &h,                                                //
                          size_t wgsize, size_t ntypes, size_t nposes,                     //
                          size_t natlig, size_t natpro,                                     //
                          const Atom *__restrict__ proteins,                                //
                          const Atom *__restrict__ ligands,                                 //
                          const FFParams *__restrict__ forcefields,                         //
                          const float *__restrict__ transforms_0, const float *__restrict__ transforms_1, //
                          const float *__restrict__ transforms_2, const float *__restrict__ transforms_3, //
                          const float *__restrict__ transforms_4, const float *__restrict__ transforms_5, //
                          float *__restrict__ energies) {

    size_t global = std::ceil(double(nposes) / PPWI);
    global = wgsize * size_t(std::ceil(double(global) / double(wgsize)));

    sycl::accessor<FFParams, 1, RW, Local> local_forcefield(sycl::range<1>(ntypes), h);

    h.parallel_for<bude_kernel_ndrange<PPWI>>(sycl::nd_range<1>(global, wgsize), [=](sycl::nd_item<1> item) {
      const size_t lid = item.get_local_id(0);
      const size_t gid = item.get_group(0);
      const size_t lrange = item.get_local_range(0);

      float etot[PPWI];
      sycl::float4 transform[PPWI][3];

      size_t ix = gid * lrange * PPWI + lid;
      ix = ix < nposes ? ix : nposes - PPWI;

      for (int i = lid; i < ntypes; i += lrange)
        local_forcefield[i] = forcefields[i];

      // Compute transformation matrix to private memory
      const size_t lsz = lrange;
      for (size_t i = 0; i < PPWI; i++) {
        size_t index = ix + i * lsz;

        const float sx = sycl::sin(transforms_0[index]);
        const float cx = sycl::cos(transforms_0[index]);
        const float sy = sycl::sin(transforms_1[index]);
        const float cy = sycl::cos(transforms_1[index]);
        const float sz = sycl::sin(transforms_2[index]);
        const float cz = sycl::cos(transforms_2[index]);

        transform[i][0].x() = cy * cz;
        transform[i][0].y() = sx * sy * cz - cx * sz;
        transform[i][0].z() = cx * sy * cz + sx * sz;
        transform[i][0].w() = transforms_3[index];
        transform[i][1].x() = cy * sz;
        transform[i][1].y() = sx * sy * sz + cx * cz;
        transform[i][1].z() = cx * sy * sz - sx * cz;
        transform[i][1].w() = transforms_4[index];
        transform[i][2].x() = -sy;
        transform[i][2].y() = sx * cy;
        transform[i][2].z() = cx * cy;
        transform[i][2].w() = transforms_5[index];

        etot[i] = ZERO;
      }

      item.barrier(sycl::access::fence_space::local_space);

      // Loop over ligand atoms
      for (size_t il = 0; il < natlig; il++) {
        // Load ligand atom data
        const Atom l_atom = ligands[il];
        const FFParams l_params = local_forcefield[l_atom.type];
        const bool lhphb_ltz = l_params.hphb < ZERO;
        const bool lhphb_gtz = l_params.hphb > ZERO;

        const sycl::float4 linitpos(l_atom.x, l_atom.y, l_atom.z, ONE);
        sycl::float3 lpos[PPWI];
        for (size_t i = 0; i < PPWI; i++) {
          // Transform ligand atom
          lpos[i].x() = transform[i][0].w() + linitpos.x() * transform[i][0].x() + linitpos.y() * transform[i][0].y() +
                        linitpos.z() * transform[i][0].z();
          lpos[i].y() = transform[i][1].w() + linitpos.x() * transform[i][1].x() + linitpos.y() * transform[i][1].y() +
                        linitpos.z() * transform[i][1].z();
          lpos[i].z() = transform[i][2].w() + linitpos.x() * transform[i][2].x() + linitpos.y() * transform[i][2].y() +
                        linitpos.z() * transform[i][2].z();
        }

        // Loop over protein atoms
        #pragma unroll 4
        for (size_t ip = 0; ip < natpro; ip++) {
          // Load protein atom data
          const Atom p_atom = proteins[ip];
          const FFParams p_params = local_forcefield[p_atom.type];

          const float radij = p_params.radius + l_params.radius;
          const float r_radij = ONE / (radij);

          const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
          const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
          const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

          const bool phphb_ltz = p_params.hphb < ZERO;
          const bool phphb_gtz = p_params.hphb > ZERO;
          const bool phphb_nz = p_params.hphb != ZERO;
          const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
          const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
          const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
          const float r_distdslv = ONE / (distdslv);

          const float chrg_init = l_params.elsc * p_params.elsc;
          const float dslv_init = p_hphb + l_hphb;

          #pragma unroll
          for (size_t i = 0; i < PPWI; i++) {
            // Calculate distance between atoms
            const float x = lpos[i].x() - p_atom.x;
            const float y = lpos[i].y() - p_atom.y;
            const float z = lpos[i].z() - p_atom.z;

            const float distij = sycl::sqrt(x * x + y * y + z * z);

            // Calculate the sum of the sphere radii
            const float distbb = distij - radij;
            const bool zone1 = (distbb < ZERO);

            // Calculate steric energy
            etot[i] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

            // Calculate formal and dipole charge interactions
            float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
            const float neg_chrg_e = -sycl::fabs(chrg_e);
            chrg_e = type_E ? neg_chrg_e : chrg_e;
            etot[i] += chrg_e * CNSTNT;

            // Calculate the two cases for Nonpolar-Polar repulsive interactions
            const float coeff = (ONE - (distbb * r_distdslv));
            float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
            dslv_e *= (zone1 ? ONE : coeff);
            etot[i] += dslv_e;
          }
        };
      }

      // Write results
      const size_t td_base = gid * lrange * PPWI + lid;

      if (td_base < nposes) {
        for (size_t i = 0; i < PPWI; i++) {
          energies[td_base + i * lrange] = etot[i] * HALF;
        }
      }
    });
  }

  std::vector<cl::sycl::device> devices;

public:
  IMPL_CLS() : devices(sycl::device::get_devices()) {}

  [[nodiscard]] std::string name() override { return "sycl"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    std::vector<Device> xs;
    for (size_t i = 0; i < devices.size(); i++)
      xs.emplace_back(i, devices[i].template get_info<sycl::info::device::name>());
    return xs;
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t deviceIdx) const override {
    auto device = devices[deviceIdx];

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();
    sycl::queue queue(device);

    // Allocate device memory (USM)
    auto *d_proteins = sycl::malloc_device<Atom>(p.protein.size(), queue);
    auto *d_ligands = sycl::malloc_device<Atom>(p.ligand.size(), queue);
    auto *d_forcefields = sycl::malloc_device<FFParams>(p.forcefield.size(), queue);
    auto *d_transforms_0 = sycl::malloc_device<float>(p.poses[0].size(), queue);
    auto *d_transforms_1 = sycl::malloc_device<float>(p.poses[1].size(), queue);
    auto *d_transforms_2 = sycl::malloc_device<float>(p.poses[2].size(), queue);
    auto *d_transforms_3 = sycl::malloc_device<float>(p.poses[3].size(), queue);
    auto *d_transforms_4 = sycl::malloc_device<float>(p.poses[4].size(), queue);
    auto *d_transforms_5 = sycl::malloc_device<float>(p.poses[5].size(), queue);
    auto *d_energies = sycl::malloc_device<float>(sample.energies.size(), queue);

    // Copy input data from host to device
    queue.memcpy(d_proteins, p.protein.data(), p.protein.size() * sizeof(Atom));
    queue.memcpy(d_ligands, p.ligand.data(), p.ligand.size() * sizeof(Atom));
    queue.memcpy(d_forcefields, p.forcefield.data(), p.forcefield.size() * sizeof(FFParams));
    queue.memcpy(d_transforms_0, p.poses[0].data(), p.poses[0].size() * sizeof(float));
    queue.memcpy(d_transforms_1, p.poses[1].data(), p.poses[1].size() * sizeof(float));
    queue.memcpy(d_transforms_2, p.poses[2].data(), p.poses[2].size() * sizeof(float));
    queue.memcpy(d_transforms_3, p.poses[3].data(), p.poses[3].size() * sizeof(float));
    queue.memcpy(d_transforms_4, p.poses[4].data(), p.poses[4].size() * sizeof(float));
    queue.memcpy(d_transforms_5, p.poses[5].data(), p.poses[5].size() * sizeof(float));
    queue.wait_and_throw();

    auto contextEnd = now();
    sample.contextTime = {contextStart, contextEnd};

    for (size_t i = 0; i < p.iterations + p.warmupIterations; ++i) {
      auto kernelStart = now();
      queue.submit([&](sycl::handler &h) {
        fasten_main(h, wgsize, p.ntypes(), p.nposes(), p.natlig(), p.natpro(),  //
                    d_proteins, d_ligands, d_forcefields,                        //
                    d_transforms_0, d_transforms_1, d_transforms_2,              //
                    d_transforms_3, d_transforms_4, d_transforms_5,              //
                    d_energies);
      });
      queue.wait_and_throw();
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    // Copy results back from device to host
    queue.memcpy(sample.energies.data(), d_energies, sample.energies.size() * sizeof(float));
    queue.wait_and_throw();

    // Free device memory
    sycl::free(d_proteins, queue);
    sycl::free(d_ligands, queue);
    sycl::free(d_forcefields, queue);
    sycl::free(d_transforms_0, queue);
    sycl::free(d_transforms_1, queue);
    sycl::free(d_transforms_2, queue);
    sycl::free(d_transforms_3, queue);
    sycl::free(d_transforms_4, queue);
    sycl::free(d_transforms_5, queue);
    sycl::free(d_energies, queue);

    return sample;
  };
};
