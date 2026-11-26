// SPDX-License-Identifier: MPL-2.0
#include "bgbot/analyzer_session.h"
#include "bgbot/pubeval.h"
#include <stdexcept>

using namespace bgbot;

int main() {
    const Board board{0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5,
                      0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0};
    // One entry check and 21 rolls per traversal. Interrupt first in the
    // cube decision, then in the independent probability search.
    struct Expired {};
    for (int limit : {4, 25}) {
        const auto base = std::make_shared<PubEval>();
        int checks = 0;
        try {
            evaluate_cube_tree(board, CubeInfo{}, base, nullptr, 2, MoveFilters::TINY,
                               1, false, nullptr, [&] {
                if (++checks == limit) throw Expired{};
            });
            throw std::runtime_error("cube search did not honor the budget");
        } catch (const Expired&) {}
    }
}
