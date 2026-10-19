# The build ID: which source a binary was built from, so that the
# coordinator of distributed self-play can refuse a remote binary built from
# other code. Run at build time (cmake -P), not at configure time, so a new
# commit or an edit changes the ID without re-running CMake.
#
#   cmake -DSRC=<source dir> -DOUT=<generated file> -DTEMPLATE=<build_info.hpp.in>
#         [-DOVERRIDE=<id>] -P azgo_build_info.cmake
#
# The ID is the commit (`git rev-parse HEAD`, 40 hex). When `git status
# --porcelain` lists anything, "+" and the first 8 hex of a SHA-256 over that
# listing and `git diff HEAD` follow, so a dirty tree never passes for its
# commit (untracked files count: a new src/*.cpp joins the build). OVERRIDE
# (the AZGO_BUILD_ID cache variable) replaces git, for copies made with `git
# archive`, which have no .git. Neither gives "unknown". OUT is rewritten
# only when the ID changed, so an untouched rebuild recompiles nothing.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).

# 40 lowercase hex, optionally "+" and 8 lowercase hex. CMake's regular
# expressions have no {n}, so the lengths are checked separately.
function(azgo_valid_build_id id out)
  string(LENGTH "${id}" length)
  if(id MATCHES "^[0-9a-f]+(\\+[0-9a-f]+)?$" AND (length EQUAL 40 OR length EQUAL 49))
    string(SUBSTRING "${id}" 0 40 commit)
    if(commit MATCHES "^[0-9a-f]+$")
      set(${out} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${out} FALSE PARENT_SCOPE)
endfunction()

set(id "unknown")
if(NOT "${OVERRIDE}" STREQUAL "")
  azgo_valid_build_id("${OVERRIDE}" ok)
  if(NOT ok)
    message(FATAL_ERROR "AZGO_BUILD_ID='${OVERRIDE}' is not a build ID (40 lowercase hex digits, "
                        "optionally followed by + and 8 more)")
  endif()
  set(id "${OVERRIDE}")
else()
  find_program(AZGO_GIT git)
  if(AZGO_GIT)
    # Only SRC's own repository: a copy unpacked inside another repository
    # must not take that repository's commit.
    execute_process(COMMAND "${AZGO_GIT}" -C "${SRC}" rev-parse --show-toplevel
                    OUTPUT_VARIABLE top RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    file(REAL_PATH "${SRC}" src_real)
    if(rc EQUAL 0)
      file(REAL_PATH "${top}" top_real)
    endif()
    if(rc EQUAL 0 AND top_real STREQUAL src_real)
      execute_process(COMMAND "${AZGO_GIT}" -C "${SRC}" rev-parse HEAD
                      OUTPUT_VARIABLE head RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      string(LENGTH "${head}" length)
      if(rc EQUAL 0 AND length EQUAL 40 AND head MATCHES "^[0-9a-f]+$")
        set(id "${head}")
        execute_process(COMMAND "${AZGO_GIT}" -C "${SRC}" status --porcelain
                        OUTPUT_VARIABLE status ERROR_QUIET)
        if(NOT "${status}" STREQUAL "")
          execute_process(COMMAND "${AZGO_GIT}" -C "${SRC}" diff HEAD
                          OUTPUT_VARIABLE diff ERROR_QUIET)
          string(SHA256 hash "${status}${diff}")
          string(SUBSTRING "${hash}" 0 8 hash)
          set(id "${id}+${hash}")
        endif()
      endif()
    endif()
  endif()
endif()

set(AZGO_BUILD_ID_VALUE "${id}")
configure_file("${TEMPLATE}" "${OUT}.tmp" @ONLY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${OUT}.tmp" "${OUT}")
file(REMOVE "${OUT}.tmp")
