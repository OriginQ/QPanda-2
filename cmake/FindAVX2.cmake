include(CheckCSourceRuns)
include(CheckCXXSourceRuns)

set(AVX2_CODE "
#include <immintrin.h>
int main(){
  __m256d a,b;
  a = _mm256_set_pd(0,0,0,0);
  b = _mm256_permute4x64_pd(a, 0);
  return 0;
}
")

macro(CHECK_SUPPORT_AVX2)
	if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
		if(APPLE OR UNIX)
				set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
				set(CMAKE_REQUIRED_FLAGS "-mfma -mavx2")
				set(AVX2_FLAGS "-mfma -mavx2")
				CHECK_C_SOURCE_RUNS("${AVX2_CODE}" C_HAS_AVX2)
				CHECK_CXX_SOURCE_RUNS("${AVX2_CODE}" CXX_HAS_AVX2)

				if(C_HAS_AVX2 AND CXX_HAS_AVX2)
					set(AVX2_FOUND TRUE CACHE BOOL "AVX2 support")
				else()
					set(AVX2_FOUND FALSE CACHE BOOL "AVX2 not support")
				endif()
				set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
		elseif(MINGW)
				set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
				set(CMAKE_REQUIRED_FLAGS "-mfma -mavx2")
				set(AVX2_FLAGS "-mfma -mavx2")
				CHECK_C_SOURCE_RUNS("${AVX2_CODE}" C_HAS_AVX2)
				CHECK_CXX_SOURCE_RUNS("${AVX2_CODE}" CXX_HAS_AVX2)

				if(C_HAS_AVX2 AND CXX_HAS_AVX2)
					set(AVX2_FOUND TRUE CACHE BOOL "AVX2 support")
				else()
					set(AVX2_FOUND FALSE CACHE BOOL "AVX2 not support")
				endif()
				set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
		elseif(MSVC)
			set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
			set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
			set(AVX2_FLAGS "/arch:AVX2")
			CHECK_CXX_SOURCE_RUNS("${AVX2_CODE}" CXX_HAS_AVX2)
			CHECK_C_SOURCE_RUNS("${AVX2_CODE}" C_HAS_AVX2)

			if(C_HAS_AVX2 AND CXX_HAS_AVX2)
				set(AVX2_FOUND TRUE CACHE BOOL "AVX2 support")
			else()
				set(AVX2_FOUND FALSE CACHE BOOL "AVX2 not support")
			endif()

			set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
		else()
			set(AVX2_FOUND FALSE CACHE BOOL "AVX2 support")
		endif()

		mark_as_advanced(AVX2_FLAGS)
		mark_as_advanced(AVX2_FOUND)
	endif()

endmacro()

