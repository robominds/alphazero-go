# The build ID: which source a binary was built from, so that the
# coordinator of distributed self-play can refuse a remote binary built from
# other code. Run at build time (cmake -P), not at configure time, so a new
# commit or an edit changes the ID without re-running CMake.
#
#   cmake -DSRC=<source dir> -DOUT=<generated file> -DTEMPLATE=<build_info.hpp.in>
#         [-DOVERRIDE=<id>] -P azgo_build_info.cmake
#
# The ID is the commit (`git rev-parse HEAD`, 40 hex). When a tracked file
# changed, or an untracked file lies where the build globs its sources
# (CMakeLists.txt, cmake, src, include, apps, tests: a new src/*.cpp joins
# the build), "+" and the first 8 hex of a SHA-256 over the status, `git diff
# HEAD` and those untracked files' names and contents follow, so a dirty
# tree never passes for its commit. Untracked files anywhere else (notes,
# run directories) leave the ID clean, so two clean checkouts of a commit
# get the same ID. OVERRIDE
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
    # --no-optional-locks: reading the status must not take the index lock
    # (or refresh the index) while another git command runs in the tree.
    set(git "${AZGO_GIT}" --no-optional-locks -C "${SRC}")
    # Only SRC's own repository: a copy unpacked inside another repository
    # must not take that repository's commit.
    execute_process(COMMAND ${git} rev-parse --show-toplevel
                    OUTPUT_VARIABLE top RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    get_filename_component(src_real "${SRC}" REALPATH)
    if(rc EQUAL 0)
      get_filename_component(top_real "${top}" REALPATH)
    endif()
    if(rc EQUAL 0 AND top_real STREQUAL src_real)
      execute_process(COMMAND ${git} rev-parse HEAD
                      OUTPUT_VARIABLE head RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      string(LENGTH "${head}" length)
      if(rc EQUAL 0 AND length EQUAL 40 AND head MATCHES "^[0-9a-f]+$")
        set(id "${head}")
        # Changes to tracked files, and untracked files where the build looks
        # for sources (the globs of CMakeLists.txt), with their contents, so
        # editing a new, still untracked source changes the ID too. Untracked
        # files elsewhere (notes, run directories) leave the ID clean.
        execute_process(COMMAND ${git} status --porcelain --untracked-files=no
                        OUTPUT_VARIABLE status ERROR_QUIET)
        execute_process(COMMAND ${git} ls-files --others --exclude-standard --
                                CMakeLists.txt cmake src include apps tests
                        OUTPUT_VARIABLE untracked ERROR_QUIET)
        if(NOT "${status}${untracked}" STREQUAL "")
          execute_process(COMMAND ${git} diff --no-ext-diff --no-color HEAD
                          OUTPUT_VARIABLE diff ERROR_QUIET)
          set(contents "")
          string(REPLACE "\n" ";" untracked_list "${untracked}")
          foreach(f IN LISTS untracked_list)
            if(NOT "${f}" STREQUAL "" AND EXISTS "${SRC}/${f}")
              file(SHA256 "${SRC}/${f}" file_hash)
              string(APPEND contents "${f} ${file_hash}\n")
            endif()
          endforeach()
          string(SHA256 hash "${status}${diff}${contents}")
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
