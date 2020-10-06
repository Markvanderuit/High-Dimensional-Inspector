/*-******************************************************* -- HEADER -{{{1- */
/*- Verbatim GLSL code
 */
/*-***************************************************************** -}}}1- */

#ifndef FLUX_GL_VERBATIM_GLSL_HPP_0847F58F_6597_4220_9B3F_DEA0A59C78E2
#define FLUX_GL_VERBATIM_GLSL_HPP_0847F58F_6597_4220_9B3F_DEA0A59C78E2

//--//////////////////////////////////////////////////////////////////////////
//--    include                     ///{{{1///////////////////////////////////

/** Verbatim GLSL code
 *
 * Embed verbatim GLSL code in C++ code. `FLUX_GL_VERBATIM_GLSL` turns its
 * argument(s) into a string literal.
 *
 * Example:
   \code
   char const* glsl = FLUX_GL_VERBATIM_GLSL(
   		out vec3 oColor;
		void main() {
			oColor = vec3( 0.3, 0.3, 0.3 );
		}
   ) );
   \endcode
 *
 * `VERBATIM_GLSL` expands macros in its argument. GLSL preprocessor must be
 * wrapped into `FLUX_GL_VERBATIM_PROTECT()` to avoid C++ preprocessor errors:
   \code
   char const* glsl = FLUX_GL_VERBATIM_GLSL(
   	   FLUX_GL_VERBATIM_PROTECT( #version 150 )
       out vec3 oColor;
  
   	   void main() { ... }
   );
   \endcode
 * produces a string that contains
   \code
   #version 150
   out vec3 oColor; void main() { ... }
   \endcode
 *
 * \warning The resulting string literal doesn't contain line breaks (other
 * than those caused by `FLUX_GL_VERBATIM_PROTECT`). Single- line GLSL comments
 * (`//`) therefore won't work as expected and should be avoided!
 */
#define FLUX_GL_VERBATIM_GLSL(...) FLUX_GL_VERBATIM_GLSL_IMPL_(__VA_ARGS__)
#define FLUX_GL_VERBATIM_GLSL_IMPL_(...) #__VA_ARGS__

/** Helper: enable GLSL preprocessor in verbatim GLSL code
 *
 * Enable use of GLSL preprocessor statements in verbatim GLSL code blocks
 * using `FLUX_GL_VERBATIM_GLSL()`:
   \code
   char const* glsl = FLUX_GL_VERBATIM_GLSL(
      FLUX_GL_VERBATIM_PROTECT( #version 150 )
      ...
   );
   \endcode
 *
 * \note `FLUX_GL_VERBATIM_PROTECT()` injects a newline before and after its
 * contents.
 *
 * \see FLUX_GL_VERBATIM_GLSL
 */
#define FLUX_GL_VERBATIM_PROTECT( ... ) \n __VA_ARGS__\n

//--///}}}1/////////////// vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
#endif // FLUX_GL_VERBATIM_GLSL_HPP_0847F58F_6597_4220_9B3F_DEA0A59C78E2
