// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/mobile.h"
#include "bgbot/analyzer_session.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/neural_net.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace bgbot;
struct NeongammonMobileEngine {
    BearoffDB db;
    std::shared_ptr<Strategy> base;
    std::shared_ptr<Strategy> strategy;
    std::shared_ptr<MultiPlyStrategy> checker_2ply;
};
namespace {
thread_local std::string last_error;

template<typename T, size_t N> void array(std::ostringstream& s, const std::array<T,N>& a) {
    s << '[';
    for(size_t i=0;i<N;++i) { if(i) s << ','; s << a[i]; }
    s << ']';
}
void validate(const NeongammonMobileRequest& r, int operation) {
    validate_met_away_scores(r.away1, r.away2);
    if(operation<0 || operation>1 || r.ply<1 || r.ply>2 || r.cube_owner<0 || r.cube_owner>2 ||
       r.cube_value<1 || r.cube_value>32768 || (r.cube_value & (r.cube_value-1)) ||
       r.away1<0 || r.away2<0 ||
       ((r.away1==0)!=(r.away2==0)) || r.budget_ms<1 || r.budget_ms>1000 ||
       (operation==0 && (r.die1<1 || r.die1>6 || r.die2<1 || r.die2>6)))
        throw std::invalid_argument("unsupported request");
    int player=r.board[25], opponent=r.board[0];
    if(player<0 || opponent<0 || player>15 || opponent>15) throw std::invalid_argument("invalid bar");
    for(int i=1;i<25;++i) {
        if(r.board[i]<-15 || r.board[i]>15) throw std::invalid_argument("invalid checker count");
        if(r.board[i]>0) player+=r.board[i]; else opponent-=r.board[i];
    }
    if(player<1 || player>15 || opponent<1 || opponent>15)
        throw std::invalid_argument("invalid or terminal board");
}
}
extern "C" NeongammonMobileEngine* neongammon_mobile_create(const char* type,
    const char* const* paths, const int* hidden, int count, const char* bearoff) {
    try {
        if(!type || !paths || !hidden || !bearoff || count<5 || count>32)
            throw std::invalid_argument("invalid model configuration");
        auto e=std::make_unique<NeongammonMobileEngine>();
        std::vector<std::string> p; std::vector<int> h;
        for(int i=0;i<count;++i) { if(!paths[i] || hidden[i]<1 || hidden[i]>4096) throw std::invalid_argument("invalid model asset"); p.emplace_back(paths[i]); h.push_back(hidden[i]); }
        const std::string t(type);
        if(t=="backgame_pair_phased") e->base=std::make_shared<BackgameAwarePairStrategy>(p,h,true);
        else if(t=="backgame_pair") e->base=std::make_shared<BackgameAwarePairStrategy>(p,h);
        else if(t=="pair") e->base=std::make_shared<GamePlanPairStrategy>(p,h);
        else if(t=="5nn") e->base=std::make_shared<GamePlanStrategy>(p,h);
        else throw std::invalid_argument("unsupported model strategy");
        if(!e->db.load(bearoff)) throw std::runtime_error("bearoff asset unavailable");
        e->strategy=std::make_shared<BearoffStrategy>(e->base,&e->db);
        const auto filter = MoveFilters::TINY;
        e->checker_2ply = make_multipy(e->base, 2, filter.max_moves, filter.threshold, false, false, 1);
        e->checker_2ply->set_bearoff_db(&e->db);
        last_error.clear(); return e.release();
    } catch(const std::exception& ex) { last_error=ex.what(); return nullptr; }
    catch(...) { last_error="engine initialization failed"; return nullptr; }
}
extern "C" void neongammon_mobile_destroy(NeongammonMobileEngine* e) { delete e; }
extern "C" void neongammon_mobile_free(char* p) { std::free(p); }
extern "C" const char* neongammon_mobile_last_error() { return last_error.c_str(); }

extern "C" char* neongammon_mobile_analyze(NeongammonMobileEngine* e, int operation,
    const NeongammonMobileRequest* request) {
    try {
        if(!e || !request) throw std::invalid_argument("engine unavailable");
        const auto& r=*request; validate(r,operation);
        Board board{}; std::copy(r.board,r.board+26,board.begin());
        CubeInfo ci{r.cube_value,static_cast<CubeOwner>(r.cube_owner),
            {r.away1,r.away2,r.is_crawford!=0},-1.0f,
            r.away1==0 && r.jacoby!=0,r.away1==0 && r.beaver!=0};
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(r.budget_ms);
        const auto check_budget=[&] { if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("native budget exceeded"); };
        std::ostringstream s; s.imbue(std::locale::classic()); s << std::setprecision(17) << std::boolalpha;
        const auto filter = MoveFilters::TINY;
        const auto life = AnalyzerSession::contact_life(r.ply, ci.match);
        if(operation==0) {
            CheckerPosition pos{board, r.die1, r.die2, ci.cube_value, ci.owner, ci.match};
            std::vector<ScoredMove> moves;
            if (r.ply == 1) {
                moves = ranked_checker_moves_1ply(pos, *e->strategy, ci.jacoby, ci.beaver, life, check_budget);
            } else {
                const auto& depths = AnalyzerSession::filtered_plies;
                AnalyzerSession session(*e->strategy, *e->checker_2ply, filter, ci.jacoby, ci.beaver,
                    AnalyzerSession::contact_life(1, ci.match), life,
                    std::find(depths.begin(), depths.end(), r.ply) != depths.end());
                const ClearSearchCacheOnExit cleanup{*e->checker_2ply};
                moves = session.score_checker_position(pos, 1, check_budget);
            }
            s << "{\"moves\":[";
            for(size_t i=0;i<moves.size();++i) {
                check_budget();
                const auto& m=moves[i]; if(i) s << ',';
                s << "{\"board\":"; array(s,m.board); s << ",\"probs\":"; array(s,m.probs);
                s << ",\"equity\":" << m.cubeful_equity << ",\"cubeless_equity\":" << m.cubeless_equity
                  << ",\"equity_diff\":" << m.cubeful_equity-moves[0].cubeful_equity << ",\"eval_level\":\"" << (m.searched ? r.ply : 1) << "-ply\"}";
            } s << "]}";
        } else {
            ci.contact_x = static_cast<float>(life.value);
            ci.contact_x_pip_slope = static_cast<float>(life.pip_slope);
            const auto result = r.ply == 1
                ? evaluate_cube_leaf(board, ci, *e->strategy)
                : evaluate_cube_tree(board, ci, e->strategy, &e->db, r.ply, filter, 1, false, nullptr, check_budget);
            const auto& p = result.probs;
            const auto& cd = result.decision;
            const char* action=!cd.should_double ? "No Double" : cd.is_beaver ? "Double/Beaver" : cd.should_take ? "Double/Take" : "Double/Pass";
            s << "{\"probs\":"; array(s,p);
            s << ",\"cubeless_equity\":" << cubeless_equity(p)
              << ",\"equity_nd\":" << cd.equity_nd << ",\"equity_dt\":" << cd.equity_dt
              << ",\"equity_dp\":" << cd.equity_dp << ",\"should_double\":" << cd.should_double
              << ",\"should_take\":" << cd.should_take << ",\"optimal_equity\":" << cd.optimal_equity
              << ",\"optimal_action\":\"" << action << "\",\"is_beaver\":" << cd.is_beaver
              << ",\"eval_level\":\"" << r.ply << "-ply\",\"cubeless_se\":null,\"equity_nd_se\":null,\"equity_dt_se\":null}";
        }
        check_budget(); auto text=s.str(); char* out=static_cast<char*>(std::malloc(text.size()+1));
        if(!out) throw std::bad_alloc(); std::memcpy(out,text.c_str(),text.size()+1);
        last_error.clear(); return out;
    } catch(const std::exception& ex) { last_error=ex.what(); return nullptr; }
    catch(...) { last_error="native evaluation failed"; return nullptr; }
}
