#ifndef XPLANE_LIB_H
#define XPLANE_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

int32_t sort(x_addr *dests, uint32_t n_outputs, x_addr src,
									idx_t keypos, idx_t reclen);

/* sort extracted key batch and ptr batch
 * @dests_k, @dests_p: output batch of keys and that of ptrs
 * @src_k, @src_p: ditto
 */
int32_t sort_kp(x_addr *dests_k, x_addr *dests_p,
								uint32_t n_outputs,
								x_addr src_k, x_addr src_p);

int32_t merge(x_addr *dests, uint32_t n_outputs, x_addr srcA, x_addr srcB,
									idx_t keypos, idx_t reclen);

/* merge two extracted key batches and the corresponding ptr batches
 *
 * @dests_k, @dests_p: output batch of keys and that of ptrs
 * @srcA_k, @srcA_p: ditto
 */
int32_t merge_kp(x_addr *dests_k, x_addr *dests_p,
							uint32_t n_outputs,
							x_addr srcA_k, x_addr srcA_p,
							x_addr srcB_k, x_addr srcB_p);
#ifdef __cplusplus
}		// extern C
#endif

#endif

