#include <cstddef>
#include <kaad/exceptions.hpp>
#include <string>
#include <string_view>

namespace kaad {

std::string make_graph_errmsg(std::size_t graph_idx, const char *op_name,
                              std::string_view msg) {

    std::string errmsg;
    const std::size_t MAX_MSG_LEN = 128;
    errmsg.reserve(MAX_MSG_LEN);

    errmsg += "in node[";
    errmsg += std::to_string(graph_idx);
    errmsg += "] (";
    errmsg += op_name;
    errmsg += "), ";
    errmsg += msg;

    return errmsg;
}

} // namespace kaad
