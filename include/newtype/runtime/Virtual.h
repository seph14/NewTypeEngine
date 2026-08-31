#pragma once

/**
 * @brief Runtime mode support macros
 *
 * When RT_RUNTIME is defined, enables virtual functions and dynamic dispatch
 * for hot-reload support. When not defined, provides zero-overhead static linking.
 */

#ifdef RT_RUNTIME

// Runtime mode: enable virtual functions
#define RT_DECL
#define rt_virtual virtual

#else

// Static mode: no virtual overhead
#define RT_DECL
#define rt_virtual

#endif
