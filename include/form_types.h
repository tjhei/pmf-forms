#ifndef PMF_FORM_FORM_TYPES_H
#define PMF_FORM_FORM_TYPES_H

namespace pmf
{
  namespace forms
  {
    /** @brief Value-shape categories represented by form expressions. */
    enum class ValueShape
    {
      scalar,          ///< A scalar value.
      vector,          ///< A vector value.
      tensor,          ///< A rank-2 tensor value.
      symmetric_tensor ///< A symmetric rank-2 tensor value.
    };

    /** @brief Marker type for cell integration. */
    struct CellMeasure
    {};

    /** @brief Cell integration measure. */
    constexpr CellMeasure dx{};
  } // namespace forms
} // namespace pmf

#endif // PMF_FORM_FORM_TYPES_H
