import 'package:bishop/bishop.dart';

final pgn = """Processing game...
FEN: 
Moves: e4 c5 c3 Nf6 e5 Nd5 d4 Nc6 Nf3 cxd4 cxd4 e6 a3 d6 Bd3 Qa5 Bd2 Qb6 Nc3 Nxc3 Bxc3 dxe5 dxe5 Be7 O-O Bd7 Nd2 Qc7 Qg4 O-O-O Rfc1 Kb8 Qc4 Rc8 b4 f6 Nf3 Qb6 Qe4 f5 Qe1 a6 Rab1 g5 Nd2 Nd4 Qe3 Rxc3 Rxc3 f4 Qe1 g4 Ne4 Bc6 Nc5 Ka7 a4 Bf3 a5 Qd8 Bc4 Bxc5 bxc5 Qh4 gxf3 gxf3 Kh1 Rg8 Qe4 Rg7 Qxd4 Qg5 c6 Kb8 c7 Rxc7 Rg1 Qh5 Rg8 Rc8 Qd6 Ka7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 Nf6 e5 Ne4 Nxe4 dxe4 Be3 b6 Ne2 Bb7 Ng3 c5 dxc5 Qxd1 Rxd1 Bxc5 Bxc5 bxc5 Bb5 Ke7 O-O Bc6 Bxc6 Nxc6 Nxe4 Nxe5 Nxc5 Rac8 b4 Rhd8 f3 a5 c3 axb4 cxb4 Nc6 Rxd8 Rxd8 b5 Nb4 Rb1 Nd5 b6 Rb8 b7 Nc7 Rb6 Kd8 Kf2 Nd5 Rd6 Kc7 Rxd5 exd5 Na6 Kxb7 Nxb8 Kxb8 Ke3 Kc7 Kd4 Kc6 a4 Kb6 Kxd5 Ka5 Kd6 Kxa4 Ke7 f5 Kf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Ndb5 Bb4 a3 Bxc3 Nxc3 d5 exd5 exd5 Bd3 O-O O-O d4 Ne4 Bf5 Bg5 Bxe4 Bxe4 h6 Bxf6 Qxf6 Qh5 Rfe8 Rfe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 e6 Nf3 d6 g3 Nc6 Bg2 Nf6 d3 Be7 O-O O-O h3 Rb8 a4 b6 Re1 Bb7 Bg5 h6 Be3 e5 g4 Nd4 Kh2 Nh7 Ng1 Bg5 Nd5 Ne6 Ne2 Bxd5 exd5 Nf4 Bxf4 Bxf4 Nxf4 exf4 Be4 g6 Qd2 Qf6 c3 Rbe8 f3 Qh4 Qg2 Ng5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 a4 g6 Bd3 Bg7 Nf3 Bg4 Be3 Nc6 O-O O-O Qe1 Bxf3 Rxf3 e6 Qh4 Qd8 Rh3 h5 Be2 d5 e5 Nd7 Qf2 Qe7 g4 hxg4 Bxg4 Rad8 Rd1 f5 Bf3 Rf7 Kh1 g5 Bh5 g4 Bxf7 Qxf7 Rg3 Qh5 Kg2 Nf8 Ne2 Ng6 h3 Nh4 Kf1 Kf7 Ng1 d4 Bc1 gxh3 Qe2 Qxe2 Nxe2 Ng6 Rxh3 Bf8 Rb3 Rd7 Rbd3 Bc5 c3 Ba7 Be3 Rd5 cxd4 Nb4 Rb3 a5 Bd2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 d4 Nf6 Nc3 cxd4 Qxd4 e5 Qd3 h6 Be3 Be7 O-O O-O Rad1 a6 Bc4 Qc7 a4 Nc5 Bxc5 Qxc5 Nd5 Nxd5 Bxd5 Rb8 Nd2 Bd7 Bb3 b5 a5 Rbc8 Nf3 Be6 Rfe1 Rfd8 Re2 Rc7 Red2 Rdc8 h3 Qb4 Bxe6 fxe6 Qb3 Rc4 Re2 Qxb3 cxb3 Rc1 Rxc1 Rxc1 Kh2 Kf7 Ne1 Bd8 Nd3 Rd1 Nb4 Bxa5 Nxa6 Rc1 b4 Bb6 b3 Rc3 Ra2 Rxb3 Rc2 Bd4 Rc7 Kf6 f3 Ra3 Rc6 Rc3 Rc7 Rxc7 Nxc7 Bc3 Nxb5 Bxb4 Na7 Kf7 Kg3 Ke8 Kg4 Kd7 f4 g6 h4 Kc7 f5 exf5 exf5 gxf5 Kxf5 Kb7 Nb5 Kb6 Nxd6 Bxd6 g4 Kc6 g5 hxg5 hxg5 Kd5 g6 Bf8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 Be6 Nc4 Rc8 Ne3 Be7 Bxf6 Bxf6 Bc4 O-O Bb3 Nd4 O-O Bg5 Ncd5 Nxb3 axb3 g6 Kh1 Bh6 Qd3 f5 exf5 gxf5 f4 Kh8 Rad1 Qh4 Qe2 exf4 Nc4 Bf7 Qd3 Bh5 Nxd6 Bxd1 Nxc8 Bh5 Nce7 f3 gxf3 Qh3 Nf4 Qh4 Qd4 Qf6 Qxf6 Rxf6 Nxh5 Rf7 Nd5 f4 Re1
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 Qe2 Qe7 d3 Nf6 Bg5 Nbd7 Nc3 Qxe2 Bxe2 h6 Bh4 g6 d4 a6 O-O-O Bg7 Rhe1 O-O Bc4 b5 Bd5 Rb8 Bc6 Nb6 a3 Bb7 Bxb7 Rxb7 Bxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 Nbd7 O-O O-O e5 dxe5 dxe5 Ng4 e6 Nde5 Qxd8 Rxd8 Nxe5 Nxe5 Nb5 c6 Nc7 Rb8 f4 Ng4 Bxg4 Bd4 Kh1 Bb6 f5 Bxc7 fxg6 fxg6 Bh6 Be5 Rad1 Rxd1 Rxd1 Bd6 Rf1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nf6 Nc3 e6 d4 cxd4 Nxd4 Bb4 e5 Nd5 Bd2 Nxc3 bxc3 Bf8 Bd3 d6 Qe2 Nd7 Nxe6 Qb6 Nc7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 Qc7 Rh4 e6 Bf4 Qa5 Bd2 Qc7 Bf4 Qa5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 b5 Nxc6 Bxc6 Bd3 Be7 e5 dxe5 fxe5 Nd7 Bxe7 Qxe7 Be4 Bxe4 Nxe4 Nxe5 Qd4 f6 Nd6 Kf8 Rhf1 Kg8 g4 h6 h4 Nf7 Qe4 Rf8 Nf5 Qe8 Nd4 e5 Nf5 h5 Rg1 Rh7 Qb7 Kh8 gxh5 Qe6 Nxg7 Qxa2 Qe7 Rg8 Qxf6 Qa1 Kd2 Qa5 Ke2 Rgxg7 Rxg7 Rxg7 Rg1
Result: 1-0
Processing game...
FEN: 
Moves: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 a5 a4 Bg7 Ng5 e6 f4 dxe5 fxe5 c5 O-O O-O c3 Nc6 Ne4 Nd7 Be3 Ne7 Bg5 cxd4 cxd4 h6 Bh4 g5 Bf2 Ng6 Nbc3 Qe7 Bc2 b6 Be3 Ba6 Rf2 Nh8 Bxg5 hxg5 Qh5 f5 Nxg5 Rf7 Bxf5 Rxf5 Rxf5 exf5 Nd5 Qe8 Qh7 Kf8 Qxf5 Kg8 Qh7 Kf8 Ra3 Rc8 Rf3 Nf6 h3 Qg6 Rxf6 Bxf6 Ne6 Ke8 Nxf6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nf3 d5 d4 Nf6 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Qc2 Nc6 Rd1 Qe7 a3 d4 Nxd4 Bxd4 exd4 Rd8 Be3 Ng4 Be2 e5 d5 Nd4 Bxd4 exd4 Qe4 Qh4 Qxd4 Re8 Rd3 Qh6 h3 Bf5 Rd2 Rad8 O-O Ne5 f4 Ng6 Qf2 b6 b4 Be4 Bd3 f5 Bxe4 fxe4 Re2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 Nd4 Nxd4 exd4 O-O Bc5 Bc4 d6 b4 Bb6 a4 a6 d3 Ne7 Qh5 O-O Bg5 Be6 Nd2 Bxc4 Bxe7 Qxe7 Nxc4 Ba7 Qd5 Rab8 Rab1 Rfd8 b5 axb5 axb5 Bc5 Ra1 Qd7 Rfb1 c6 bxc6 bxc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 c4 d5 exd5 cxd5 cxd5 Nf6 Nc3 Nxd5 Nf3 Nc6 d4 Bg4 Qb3 Bxf3 gxf3 e6 Qxb7 Nxd4 Bb5 Nxb5 Qc6 Ke7 Qxb5 Qd7 Qa5 Nxc3 Qxc3 f6 Be3 Kf7 O-O Be7 Rac1 Qd5 Rfd1 Qxf3 Rd7 Qg4 Kf1 Rhd8 Rc7 Kf8 Bxa7 Rd5 Qg3 Rd1 Rxd1 Qxd1 Kg2 Qd5 Qf3 Qg5 Qg3 Qd5 Qf3 Qg5 Kf1 Rd8 a4 Qa5 Qg3 e5 Qc3 Qxa4 h3 Qe4 Qb3 Rd5 Be3 g5 Qb8 Kf7 Qb3 Qf3 Rc1 Qxh3 Ke1 Qf3 Rc7 g4 Qc2 Kf8 Rc8 Bd8 Rb8 g3 Bc5 Ke8 fxg3 e4 Rb7 Qh1 Ke2 Qg2 Bf2 Qf3 Ke1 e3 Bg1 Qxg3 Kf1 Qf4 Ke1 Ba5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be2 Be7 O-O Nf6 Be3 a6 f4 Bd7 Qe1 b5 a3 O-O Qg3 Qc7 Kh1 Rab8 e5 dxe5 fxe5 Nxe5 Rxf6 Bxf6 Bf4 Qb6 Bxe5 Bxe5 Qxe5 b4 axb4 f6 Qc5 Qxb4 Qxb4 Rxb4 Nb3 Bb5 Kg1 Bxe2 Nxe2 Rc8 Nc3 Rc6 Kf2 Kf7 Ra2 g5 Nc1 Rf4 Ke2 Rg4 g3 h5 Nd3 Rgc4 Ke3 h4 Ne4 hxg3 hxg3 Rb6 c3 Rc8 Ndc5 f5 Nxg5 Kg6 Nd7 Rxb2 Rxb2 Rxc3 Kd4 Rxg3 Nxe6 f4 Nxf4 Kf5 Nd3 a5 Rb5 Ke6 N7c5 Kf7 Rb7 Kg8 Ne6 a4 Ndf4 a3 Ra7 a2 Rxa2 Rg1 Ke5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Bg7 Be3 d6 Nc3 Nh6 Be2 O-O Qd2 Ng4 Bxg4 Bxg4 O-O Qa5 Rac1 Rfc8 b3 a6 Nxc6 Rxc6 Bh6 Rac8 Bxg7 Kxg7 f4 f6 Kh1 b5 f5 g5 Qf2 bxc4 Nd5 cxb3 Rxc6 Rxc6 Qa7 Qd8 axb3 Be2 Re1 Bb5 h4 gxh4 Qf2 Kh8 Qxh4 Rc2 Re3 Qf8 Rh3 Qg7 Rg3 Qf8 Nf4 Be8 Ne6 Qf7 Kg1 Qh5 Qxh5 Bxh5 Rg7 Re2 Rxe7 Rxe4 Ra7 h6 Ra8 Kh7 Ra7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 Qc7 Qe2 Ngf6 Bd2 e6 c4 Bd6 Nf5 O-O Nxd6 Qxd6 Bc3 b5 cxb5 cxb5 Qxb5 Nd5 Ne5 Nxe5 dxe5 Nxc3 bxc3 Qc7 Rh3 Rfd8 Qe2 Rd5 f4 Rad8 Re3 Qb6 Rc1 Rb5 Kf2 Rb2 Rc2 Rb1 Kg3 Rbd1 Rb2 Qc5 Qf3 a5 Qe4 Qa3 Rc2 Rh1 Kg4 Qc5 Rh3 Rhd1 a4 Rb8 Rf3 Rh1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 d4 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 h6 Bf4 e6 dxe6 Bxe6 Bxd6 Re8 Nf3 Nc6 O-O Qa5 Nd2 Red8 Nb3 Qb6 Na4 Qb4 Nbxc5 Bxc4 Bxc4 Qxc4 Nxb7 Nxe4 Rc1 Qb5 Nxd8 Rxd8 Qc2 Nd4 Qxe4 Ne2 Kh1 Nxc1 Rxc1 Rxd6 Qc2 Qg5 Rd1 Qf5 Qc1 Rxd1 Qxd1 Qxf2 Qg1 Qc2 b3 Qxa2 Nc5 Qd2 Qb1 Bd4 Nd3 Qe3 Nb4 h5 Qd1 h4 Nc2 Qxb3 Qxd4 Qb1 Qg1 Qxc2 Qxa7 Qd1 Qg1 Qxg1 Kxg1 Kg7 Kf2 Kf6 Ke3 Ke5 Kf3 f5 Ke3 g5 h3 Kd5 Kd3 Kc5 Kc3 g4 Kd3 gxh3 gxh3 Kd5 Ke3 Ke5 Kf3 f4 Kf2 Ke4 Ke2 f3 Kf1 Kf5 Kg1 Ke5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 Nf3 Nd7 Bc4 e6 Qe2 Qc7 O-O O-O-O c3 Ngf6 Re1 Be7 a4 Nd5 a5 a6 Bd2 Rhe8 b4 f6 Bd3 Bf8 Bxg6 hxg6 Nh4 g5 Ng6 Bd6 Qg4 Nf8 Nxf8 Bxf8 Rxe6 Qd7 Rae1 Bd6 h3 Rxe6 Qxe6 Qxe6 Rxe6 Bxg3 fxg3 Kd7 Re1 b5 Kf2 Ne7 h4 gxh4 gxh4 Rh8 g3 g5 Rh1 Nf5 h5 Nd6 Kf3 Ke6 g4 Nc4 Re1 Kf7 Re2 Rd8 Be1 Rd5 Bg3 Rd7 Ke4 Re7 Kd3 Rd7 h6 Rd8 h7 Rh8 Rh2 Ke7 d5 cxd5 Kd4 Kf7 Bc7 Ke6 Rh6 Ke7 Kxd5 Ne3 Kc6 Nxg4 Rh5 Ne3 Bb6 Nf5 Bc5 Ke6 Kb7 Kd7 Kb6 Nd6 Rh6 Ne4 Bd4 g4 Bxf6 Nxf6 Rxf6 Rxh7 Rg6 Re7 Rxg4 Re6 Kb7 Ke7 Rg5 Rd6 Rc5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 b3 g6 Bb2 Bg7 c4 O-O g3 d6 d4 Nbd7 Bg2 e5 e3 Re8 O-O e4 Nfd2 Nf8 Nc3 Bf5 f3 exf3 Qxf3 c6 e4 Bg4 Qf2 Qa5 c5 Be6 h3 dxc5 e5 cxd4 exf6 Bh6 Qxd4 Rad8 Nc4 Bxc4 Qxc4 Qb6 Kh1 Rd4 Na4 Rxc4 Nxb6 Rc2 Nc4 Ree2 Bf3 Rh2 Kg1 b5 Be4 Rce2 Bf3 Rc2 Be4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 Qc7 Bd2 Ngf6 Qe2 e6 O-O-O c5 Rh4 Rc8 Nf5 cxd4 N3xd4 Qc4 Qxc4 Rxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d6 Nc3 Bg4 e4 Nbd7 e5 Ng8 h3 Bxf3 Qxf3 c6 Bf4 d5 e6 fxe6 Bd3 Ngf6 Qe2 g6 Qxe6 Bg7 O-O Nh5 Bg5 Nf8 Qg4 Nf6 Qe2 Qd6 Rae1 e6 Na4 Kf7 b4 b6 Qd2 Re8 Bf4 Qe7 b5 Qa3 Nc3 c5 Nb1 Qa4 dxc5 bxc5 c4 N8d7 Nc3 Qa5 Qc2 Qd8 Bg5 Nb6 a4 dxc4 Be4 Re7 a5 Nbd7 Bc6 Nb8 Rd1 Qxa5 Ne4 Rf8 Bf4 Nxc6 bxc6 Ne8 Rd7 Rxd7 cxd7 Nf6 Nd6 Ke7 Nxc4 Qa6 Bd6 Kxd7 Bxf8 Bxf8 Qd3 Ke7 Rd1 Nd5 Qe4 Kf7 Ne5 Kg8 Nd7 c4 Rb1 Qd6 Rb7 c3 Nxf8 Kxf8 Rxh7 Qf4 Qxf4 Nxf4 Kf1 a5 Ra7 Nd5 Rxa5 Kf7 g3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Nbd2 O-O e3 b6 Bd3 Bb7 O-O d5 a3 Bxd2 Bxd2 Nbd7 cxd5 Bxd5 b4 c5 Rc1 cxd4 Nxd4 Ne5 Ba6 Ne4 Be1 Qg5 f4 Qg6 fxe5 Nc5 Bg3 Nxa6 Nf5 Rae8 Nd6 Re7 Rf4 h5 e4 Ba8 Bh4 Rd7 Rc3 Qh6 Qf1 Nc7 Rcf3 f5 exf6 Rxd6 f7 Kh7 Be7 e5 Bxf8 exf4 Bxd6 Qxd6 Qd3 Qe7 Qc4 Kh6 Rxf4 Ne6 Qc8 Qd6 Qh8 Kg6 f8=N Nxf8 Qxf8 Qd1 Kf2 Qd2 Kg3 Qe1 Kh3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 g6 Nc3 Bg7 e4 O-O Nf3 d6 Be2 e6 O-O exd5 exd5 Re8 h3 Bf5 Bd3 Ne4 Nxe4 Bxe4 Bxe4 Rxe4 Qc2 Re8 Bg5 Qd7 Rae1 Na6 a3 Nc7 b3 b5 Rxe8 Rxe8 Re1 Rxe1 Nxe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 h5 Bh7 Nf3 Nd7 Bd3 Bxd3 Qxd3 Qc7 Bd2 Ngf6 O-O-O e6 Ne4 O-O-O g3 c5 Bf4 c4 Qe2 Qc6 Nxf6 gxf6 d5 exd5 Nd4 Qa6 Kb1 Bd6 Qf3 Bxf4 Qxf4 Ne5 Qf5 Kb8 f4 Nd7 Qxd5 Ne5 Qe4 Ng4 Qe2 Qb6 c3 f5 Rhe1 Qc5 Qe7 Qxe7 Rxe7 Rhe8 Rde1 Rxe7 Rxe7 Nf6 Rxf7 Nxh5 Nxf5 Rd3 Rf8 Kc7 Rg8 Kd7 Rg6 b5 a3 Rd1 Ka2 Rg1 Rd6 Kc7 Rxh6 Nxg3 Nd4 Ne4 Nxb5 Kd7 Rh7 Ke6 Rxa7 Nc5 Nd4 Kd6 Nf5 Kd5 Ne3 Ke4 Rc7 Nd3 Re7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 Qc7 Bd2 Ngf6 O-O-O e6 Ne4 O-O-O g3 Nxe4 Qxe4 Be7 Kb1 Rhe8 Qe2 Bd6 Rhe1 Nf6 Ne5 c5 dxc5 Bxe5 Qxe5 Qxe5 Rxe5 Rd4 Kc1 Red8 f3 Nd7 Ree1 Nxc5 Bc3 Rxd1 Rxd1 Rxd1 Kxd1 f6 Bb4 Nd7 Ke2 Kd8 g4 Ke8 f4 Kf7 Bd2 g6 f5 g5 Bc3 Nb6 Bd4 Nc8 Kf3 b6 fxe6 Kxe6 c4 Ne7 Bc3 f5 Bg7 Ng8 b4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 c3 d5 exd5 exd5 d4 a6 Be2 c4 Bf4 Bd6 Bxd6 Qxd6 b3 cxb3 axb3 Ne7 O-O O-O c4 Nbc6 c5 Qf6 Nc3 Bf5 Qd2 Be4 Rfd1 Nf5 Ra4 Rad8 Ne1 Nfe7 b4 Rfe8 b5 axb5 Nxb5 Ng6 Ra3 Nf4 Bf1 h5 Nd6 Re7 Rb3 h4 f3 Bg6 Rxb7 Rxb7 Nxb7 Rb8 Na5 Nxa5 Qxa5 Rb2 Qa1 Ne2 Kh1 Ng3 hxg3 hxg3 Qa8 Kh7 Qc8 Rf2 Qh3 Kg8 Qc8 Kh7 Qh3 Kg8 Kg1 Qf4 Qc8 Kh7 Qh3
Result: 1-0
Processing game...
FEN: 
Moves: g3 d5 Nf3 c6 Bg2 Bg4 O-O Nd7 d3 Ngf6 Nbd2 e6 h3 Bh5 b3 Bd6 Bb2 O-O c4 a5 a3 Bg6 e4 dxe4 dxe4 Qc7 Qe2 e5 Nh4 Rfe8 Rad1 Nf8 Nf5 Bc5 Kh2 Ne6 Nf3 Bh5 Qc2 Bxf3 Bxf3 g6 Nh6 Kg7 Ng4 Nd4 Bxd4 Nxg4 Bxg4 Bxd4 h4 Qe7 a4 Rad8 Kg2 h5 Bh3 Qc5 Qe2 Rd6 Rd2 Qb4 Rd3 b5 axb5 cxb5 cxb5 Qxb5 Rfd1 Rf6 f3 Re7 Rc1 Rd6 Rc4 a4 Rxa4 Ba7 Rd2 Qb6 Ra1 Rxd2 Qxd2 Qxb3 Kh2 Bd4 Ra2 Qxf3 Bg2 Qb3 Ra6 Re8 Qa2 Qd1 Qa4 Qg1 Kh3 Rb8 Qc4 Rb1 Qd3 Qc1 Rc6 Qb2 Qf3 Qb7 Qf6 Kh7 Qd6 Qb3 Rc8 Qe6 Qxe6 fxe6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Bc4 Ngf6 Ng5 e6 Qe2 Nb6 Bd3 h6 N5f3 c5 dxc5 Nbd7 b4 b6 Nd4 Nxc5 Bb5 Ncd7 a3 Bb7 Ngf3 a6 Bd3 Be7 Bb2 O-O O-O Re8 Bc4 Bf8 Rad1 Qc7 Bb3 b5 c4 bxc4 Qxc4 Qxc4 Bxc4 Bd5 Bxd5 Nxd5 Nb3 Be7 g3 Rec8 Rc1 Kf8 Na5 Bf6 Bxf6 gxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nf3 Nf6 Nc3 d5 d4 Be7 Bg5 O-O e3 h6 Bh4 b6 Qb3 Bb7 Bxf6 Bxf6 cxd5 exd5 Rd1 Re8 Bd3 c5 dxc5 Nd7 c6 Bxc6 O-O Nc5 Qa3 d4 Nxd4 Bxd4 exd4 Qg5 d5 Bd7 Be2 a5 Bf3 Qf4 Ne2 Qb4 Nd4 Qxa3 bxa3 Ne4 h4 Rac8 Rfe1 Nd6 Rxe8 Rxe8 Rb1 Nc4 Rc1 Nd2 Be2 Re4 Rc7 Bg4 Bxg4 Rxd4 Be2 Rxd5 f3 Rc5 Rxc5 bxc5 Bd3 c4 Bc2 c3 Kf2 Nc4 Ke2 Nxa3 Kd3 Nb5 a4 Nc7 Kxc3
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 d5 Nb8 c4 c6 axb5 axb5 Rxa8 Bxa8 dxc6 b4 Ba4 Nxc6 Nf1 Qb8 g4 Rc8 Ng3 Nd8 g5 hxg5 Nxg5 Rxc4 Bb3 Rd4 Qc2 Nd7 Be3 Nc5 Bxd4 exd4 Bd5 Be7 h4 Nde6 Bxe6 fxe6 Qc4 d3 Nxe6 d2 Rd1 d5 exd5 Nxe6 Qe4 Bc5 Qxe6 Kh8 Kg2 Qf4 Qc8 Kh7 Qxc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bg5 O-O e3 h6 Bh4 Nbd7 Qc2 b6 cxd5 exd5 Bd3 Bb7 O-O c5 Rad1 Rc8 dxc5 Rxc5 Qb1 Re8 Nd4 a6 Bf5 Rc4 Bg3 Bb4 Nde2 Nc5 a3 Bxc3 Nxc3 Nce4 Nxe4 Nxe4 Bxe4 Rcxe4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 d5 Nc4 a4 Bd7 b3 Na5 axb5 axb5 Nbd2 Nb7 Bb2 g6 c4 Nh5 cxb5 Bxb5 Nc4 f6 Qd2 Rfb8 Bc3 Bf8 Nh2 Nf4 Ng4 Bg7 h4 Qc8 Nge3 Bxc4 bxc4 Bh6 Rxa8 Rxa8 Bd1 Ra3 g3 Nh5 Qb2 Qa8 Bxh5 Bxe3 Rxe3 gxh5 Kg2 Nd8 Qe2 Nf7 Qxh5 Qa6 Rf3 Kg7 Qf5 Qb7 Qg4 Kf8 Qf5 Qe7 Qxh7 Ra4 Bd2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Be2 b5 Bf3 Bb7 O-O Nc6 Nxc6 dxc6 e5 Qxe5 Re1 Qc7 Bh5 Be7 Rxe6 g6 Re1 Rd8 Qf3 c5 Bf4 Qb6 Qg3 gxh5 Bc7 Qg6 Bxd8 Qxg3 hxg3 Kxd8 Rad1 Kc7 Nd5 Bxd5 Rxd5 h6 Rxh5 Rh7 Rhe5 Kd7 R5e3 Rg7 Rd3 Kc7 Ra3 Rg6 Rf3 Bf6 c3 Kd7 Rd3 Kc7 Re8 Ne7 Red8 Nc6 R8d7 Kb6 Rxf7 Be7 Re3 Bd6 f4 c4 Kh2 Bc5 Re2 b4 Re4 bxc3 bxc3 Bf2 Rxc4 Bxg3 Kh3 Be1 a4 Na5 Rb4 Kc5 Rf5
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 e3 Bb4 Qc2 O-O Nd5 Re8 Qf5 Be7 Nxe5 Nb4 Nxb4 Bxb4 a3 d5 Qc2 Rxe5 axb4 Bf5 d3 Qe7 Bd2 d4 Be2 dxe3 fxe3 Bg4 Bf3 Bxf3 gxf3 Rxe3 Bxe3 Qxe3 Qe2 Qg5 Qd2 Qh4 Kd1 Nh5 Kc2 h6 Rhe1 Qf6 Ra5 Nf4 Re4 Ne6 Qe3 b6 Rd5 c5 f4 cxb4 d4 a5 Qd2 a4 Qxb4 Qg6 Qe1 Nxf4 Re5 a3 Qg3 Qc6 d5 Nxd5 bxa3 Nf6 Rd4 Rc8 Kb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 Bg4 dxe5 Qxd1 Rxd1 fxe5 Rd3 Bd6 Nbd2 Nf6 Nc4 O-O Nfxe5 Be2 Re3 Bxc4 Nxc4 Bc5 Rf3 Nxe4 Be3 Rxf3 gxf3 Nd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Be7 cxd5 Nxd5 Bd3 O-O h4 Nc6 Qc2 f5 a3 b6 O-O Kh8 Re1 Nf6 Qa4 Bd7 Ng5 Ng4 f3 Nce5 Qd1 h6 fxg4 Nxd3 Nxe6 Bxe6 Qxd3 Bg8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 a4 Nc6 Be2 e6 O-O Be7 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Rb8 Bf2 Bf8 Re1 Nd7 Qe2 Nxd4 Bxd4 b6 e5 dxe5 fxe5 Bb4 Rad1 Nf8 Rf1 Ng6 Ne4 Nxe5 Bh5 Ng6 Ng5 e5 Bxg6 hxg6 Rxf7 Qc6 Bxe5 Rb7 Qf2 Bg4 Rxb7 Qxb7 Qh4 Bh5 Qc4 Kh8 Rf1 Bc5 b4 Bf8 Rf7 Qc8 Qf4 a5 h4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 f4 a6 Qf3 Qb6 Nb3 Qc7 Bd3 b5 g4 b4 Ne2 Bb7 g5 Nfd7 Bd2 Nc6 Nbd4 Nc5 Nxc6 Qxc6 Nd4 Qd7 O-O-O Qa4 Kb1 b3 Nxb3 Nxe4 Qf1 g6 Be1 Bg7 h4 O-O h5 Nc5 Rh4 Nxd3 Rxd3 Be4 hxg6 fxg6 Rc3 Rac8 Rh2 Bf5 Qf2 Bxc3 Bxc3 e5 fxe5 Bd3 Qe3 Rf1 Nc1 Rxc3 bxc3 Qb5 Ka1 Bc4 exd6 Bf7 Rf2 Rxf2 Qxf2 Qxg5 Kb2 h5 Qd4 h4 Nd3 h3 d7 Qd8 Ne5 h2 Nc6 Qxd7 Qxd7 h1=Q Ne5 Qf1 Nxf7 Qxf7 Qc8 Kg7 Qxa6 g5 a4 g4 Qb5 Qg6 a5 g3 a6 g2 Qb7 Kh6 a7 g1=Q a8=Q Q6b6 Qxb6 Qxb6 Ka2 Qe6 Ka1 Qc4 Qf8 Kg6 Qb4 Qf1 Kb2 Kh7 Qe4 Kg8 Qd5 Kh8 Qh5 Kg8 Qg5 Kh8 Qd8 Kh7 Qd7 Kg8 Qe8 Kg7 Qe7 Kg8 Qd8 Kh7 Qd7 Kh8 Qd4 Kg8 c4 Qf8 Qd5 Kh8 Qe5 Kh7 Qh5 Kg8 Qd5 Kh7 c5 Qb8 Kc3 Kh8 Qd4 Kg8 c6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 e6 Bg2 Ne7 e4 c5 Nge2 Nbc6 d3 d6 O-O O-O Rb1 a6 a3 b5 cxb5 axb5 b4 cxb4 axb4 e5 Bg5 Qb6 Nd5 Nxd5 exd5 Nd4 Nxd4 exd4 Qb3 Bd7 Bd2 Rfc8 Rfc1 Rxc1 Rxc1 Rc8 Rxc8 Bxc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 d4 Bg7 e4 d6 Nf3 O-O Be2 Bg4 Be3 Nfd7 Ng1 Bxe2 Ngxe2 e5 O-O a5 Qd2 Nc6 f3 exd4 Nxd4 Nc5 Rad1 Ne6 Ndb5 Re8 Qc1 Qb8 Bh6 Bh8 Nd5 Nb4 a3 Na6 f4 c6 f5 cxd5 fxe6 Rxe6 exd5 Re7 Bf4 Rd7 Nxd6 Qd8 Nb5 Nc5 Qe3 b6 b4 axb4 axb4 Na6 Bg5 Qb8 d6 Nxb4 Be7 Qb7 Rxf7 Kxf7 Rf1 Bf6 Bxf6 Kg8 Qe6 Rf7 Be7 Rf8 Rxf7 Rxf7 d7 Nc6 Nd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 exd5 Bg5 Be7 Bxf6 Bxf6 g3 O-O Bg2 Re8 O-O Na6 b4 c5 Rb1 cxd4 Nb5 Nc7 Nbxd4 Qd7 Qd3 Ne6 e3 a5 Rfc1 axb4 Rxb4 Nc5 Qb1 Ne4 Rxb6 Ba6 Rxa6 Rxa6 Bf1 Rxa3 Bb5 Qd8 Bxe8 Qxe8 Qb5 Ra8 Qxd5 Rd8 Qf5 g6 Qf4 Bg7 Kg2 Nf6 Nc6 Ra8 Nfe5 Qe6 Rd1 Nh5 Qf3 Bxe5 Nxe5 Re8 Nd7 Kg7 Nc5 Qf6 g4 Qxf3 Kxf3 Nf6 g5 Ng8 Rd7 h6 h4 hxg5 hxg5 Re5 Ne4 Re7 Rd6 Re5 Kf4 Rf5 Kg4 Re5 Nd2 Re7 Kf4 f6 Ne4 fxg5 Nxg5 Nf6 e4 Nh5 Ke3 Ra7 e5 Ra3 Ke4 Ra4 Rd4 Rxd4 Kxd4 Nf4 Ke4 Ng2 Kd5 Nh4 Ne6 Kf7 Nd4 Ke7 e6 Ng2 Nc6 Ke8 Kd6 Nh4 Nd4 g5 Ke5 g4 Kf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Bd6 c5 Be7 b4 b6 Ng3 c6 f4 Ne8 Bd3 f5 O-O
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nf3 d5 d4 Nf6 Nc3 Be7 Bg5 O-O e3 Nbd7 Qc2 c5 O-O-O Qa5 Kb1 h6 h4 dxc4 Bxc4 Nb6 Bxf6 gxf6 Be2 cxd4 exd4 Bd7 Rh3 Na4 Rg3 Kh8 Qd2 Nxc3 bxc3 Kh7 Bd3 f5 Ne5 Bb5 Rf3 f6 Nc4 Qc7 Qe2 Bxc4 Bxc4 e5 Rxf5 Ba3 Qe4 Kh8 Rh5 Qh7 Qxh7 Kxh7 dxe5 Kg6 g4 fxe5 Rd7 Rae8 Rxb7 Rxf2 Rxa7 Bf8 Ra6 Kh7 Rf5
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 Nf3 O-O Be2 e5 O-O Nc6 Be3 Re8 dxe5 dxe5 h3 Be6 c5 a6 Qa4 Qe7 Rfd1 Rad8 Bc4 Nd4 Nxd4 exd4 Bxd4 Nxe4 Nxe4 Rxd4 Rxd4 Bxd4 Bxe6 Qxe6 Qxd4 Qxe4 Rd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 d4 c5 Nf3 cxd4 Nxd4 Nc6 e3 Nf6 Be2 O-O O-O Nxd4 exd4 d5 Bf4 dxc4 Bxc4 Bg4 Qd2 Rc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Nc6 O-O Be7 Be3 O-O f4 a6 Qe1 Nxd4 Bxd4 b5 Rd1 Qc7 e5 dxe5 fxe5 Nd7 Ne4 Bb7 Nf6 Kh8 Qh4 h6 Qh3 Rad8 Qd3 Nxf6 exf6 Bxf6 Rxf6 e5 Qf5 exd4 Bd3 g6 Qf2 Kg7 Rf1 Qe7 a4 bxa4 Bc4 Rd7 Be6 Rd6 Rxf7 Rxf7 Qxf7 Qxf7 Rxf7 Kg8 Rf6 Kg7 Rf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 d5 Bd7 b3 Nb7 c4 b4 Nbd2 Ne8 Nf1 g6 Bh6 Ng7 g4 f6 Ng3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 O-O e4 Nxc3 bxc3 c5 O-O h6 Bf4 cxd4 cxd4 Nc6 d5 Na5 dxe6 fxe6 Bg3 Rc8 Ne5 Bf6 Qg4 Qe8 Ng6 Bxa1 Nxf8 Bb2 Nxe6 h5 Qe2 Bf6 Nc7 Qf7 e5 Qxc7 Qxh5 Qc6 f3 Be7 Bh7 Kf8 Qf5 Ke8 Bg6 Kd8 Rd1 Qd5 Rxd5 Bxd5 Qd3 Rc1 Kf2
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 d4 Bg7 Nf3 d6 Nc3 Nf6 g3 O-O Bg2 c6 O-O Qa5 h3 e5 d5 cxd5 cxd5 Bd7 Nd2 b5 a3 Rc8 Nce4 Qc7 Nxf6 Bxf6 Ne4 Be7 f4 exf4 Bxf4 Bf5 Nf2 Qb6 e4 Bd7 e5 dxe5 Bxe5 Bc5 b4 Be3 Qf3 Re8 Bb2 Bf5 Kh1 Nd7 Ng4 h5 Nxe3 Qxe3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 f4 a6 Be2 Be7 O-O O-O Kh1 Nc6 Be3 Bd7 Qe1 b5 a3 Qb8 Qg3 b4 axb4 Qxb4 Rfd1 Qxb2 Nxc6 Bxc6 Bd4 Qb7 Rab1 Qc7 e5 dxe5 Bxe5 Qc8 Rd4 g6 Rc4 Qe8 f5 exf5 Rb6 Rc8 Rxa6 Ne4 Nxe4 Bxe4 Bd4 Qb5 Rxc8 Qb1 Bg1 Rxc8 Qe5 Bc5 Ra1 Qxc2 Bxc5 Qxe2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 e6 g3 d5 Bg2 e5 Nf3 d4 O-O Nc6 e3 Bc5 exd4 exd4 Nbd2 O-O Nb3 Qb6 Bf4 Re8 Nxc5 Qxc5 Nd2 h6 Nb3 Qxc4 Rc1 Qa4 Nc5 Qxa2 Qd2 Bf5 Ra1 Qc4 Rfc1 Qb5 Nxb7 Be4 Nd6 Qd5 f3 Bg6 Nxe8 Rxe8 Ra6 Re6 b4 d3 Be3 Nd4 Rxe6 Ne2 Kf2 Qxe6 Re1 Qf5 Bf4 Nd5 Rxe2 dxe2 Bd6 Qe6 Bc5 Bd3 Bxa7 Bb5 Bc5 h5 h4 Qf5 Ke1 Qb1 Kf2 Qf1
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 Bd7 Nf1 Nc4 d5 Nb6 g4 h5 gxh5 Bxh3 N3h2 Bxf1 Nxf1 Qd7 Ng3 Qh3 Qf3 g6 Bd1 Rfe8 Qg2 Qxg2 Kxg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 c5 d5 Ba6 Qc2 exd5 cxd5 d6 Nc3 Nbd7 Bf4 Be7 g3 O-O Bg2 Re8 O-O Nh5 Bd2 Nhf6 Rfe1 Bf8 a4 Ng4 Nb5 Bb7 e4 a6 Na3 Rb8 h3 Ngf6 Bc3 Qc7 Nd2 Bc8 Bf1 g5 Nf3 h6 Nc4 b5 axb5 axb5 e5 Nxd5 Nxd6 Bxd6 exd6 Qd8 Ne5 Nb4 Qd2 Nxe5 Rxe5 Rxe5 Bxe5 Nc6 Qe3 Nxe5 Qxe5 c4 Bg2 Be6 Ra7 b4 Be4 c3 Bh7 Kxh7 Qxe6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 d4 c5 d5 Nf6 e4 d6 Bd3 O-O f4 a6 Nge2 b5 cxb5 axb5 Nxb5 Ba6 Nec3 c4 Bc2 Bxb5 Nxb5 Qa5 Nc3 Nfd7 Qf3 Na6 O-O Nb4 Qe2 Nc5 Be3 Nbd3 Bxd3 Nxd3 e5 dxe5 fxe5 Bxe5 Bh6 Rfb8 Qf3 f5 g4 Rxb2 gxf5 Qxc3 fxg6 Qd4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Be7 cxd5 exd5 g3 a5 Bg2 Re8 O-O c6 h3 Bd6 g4 h6 f4 Na6 Ng3 b5 Qf3 b4 axb4 axb4 Nce2 Ne4 Nxe4 dxe4 Qf2 g6 Ng3 f5 b3 Be7 gxf5 gxf5 Qe2 Rf8 Qh5 Qe8 Qxh6 Rf6 Qh5 Qxh5 Nxh5 Rg6 Kh2 Be6 Bd2 Kh7 Rg1 Kh6 Ng3 Rgg8 Ra5 c5 Rga1 Nc7 dxc5 Rad8 Bxb4 Rd3 Be1 Rxe3 Ra6 Nxa6 Rxa6 Rg6 Bf2 Rxb3 Bxe4 fxe4 f5 Rbxg3 Bxg3 Bxf5 c6 Kg5 Ra7 Bc5 Rc7 Bb6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 b6 a3 Ba5 Rb1 Na6 Bd2 O-O d5 d6 Nf4 Bxc3 Bxc3 e5 Ne2 Ne4 f3 Nxc3 Nxc3 Nc7 Qd2 f5 f4 Bd7 Be2 exf4 exf4 Qh4 g3 Qh3 Kf2 Ne8 Bf1 Qh6 Kg2 Nf6 Bd3 Rae8 Rhe1 Qh5 Rxe8 Rxe8 Re1 Kf8 b4 a6 bxc5 bxc5 Rxe8 Qxe8 Nd1 Qb8 Qc3 Kf7 Ne3 g6 Nd1 Ba4 Nf2 Qb6 h3 Bd7 Kf3 Qd8 g4 Qc8 g5 Nh5 Qb2 Ke7 Bc2 Qd8 Qc3 Kf7 Nd1 Qf8 Qa5 Ke7 Qe1 Kd8 Qa5 Ke7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Bg4 Qa4 Nc6 Ne5 cxd4 Nxc6 bxc6 cxd4 O-O Rc1 Bd7 Rc5 Qb8 Bd3 e5 Bc2 exd4 Bxd4 Qd6 Be3 Be5 O-O Qc7 f4 Bd6 Ra5 c5 Qa3 Rfc8 Qc3 Bf8 e5 Rab8 f5 Bg7 fxg6 hxg6 Rxc5 Qxe5 Qxe5 Bxe5 Bxg6 Bxh2 Kxh2 fxg6 Rg5 Rc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5 Nc6 fxe6 fxe6 Nxc6 bxc6 e5 dxe5 Bxf6 gxf6 Ne4 Be7 Be2 h5 Rb3 Qa4 Nxf6 Bxf6 c4 Be7 O-O Ra7 Rb8 Rc7 Qd3 Bc5 Kh1 Ke7 Qg6 Kd6 Rd1 Bd4 Rxd4 exd4 Qg3 e5 c5 Kd5 Bf3 e4 Qg5 Kc4 Qc1 Kd5 Bd1 Qxa2 Qg5 Kc4 Qc1 Kd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 b3 g6 Bb2 Bg7 c4 O-O e3 c5 Be2 b6 O-O Bb7 d4 cxd4 Nxd4 d5 Nc3 Nc6 Nxd5 Nxd5 cxd5 Nxd4 Bxd4 Bxd4 Qxd4 Qxd5 Qxd5 Bxd5 Rfd1 Rfd8 f3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 e4 b5 e5 Nd5 a4 e6 axb5 Nxc3 bxc3 cxb5 Ng5 Bb7 Qh5 Qd7 Nxh7 Nc6 Nxf8 Rxh5 Nxd7 Kxd7 Rb1 a6 Bxc4 Na5 Bf1 Be4 Rb4 Bd5 f3 Rc8 Bd2 Bc4 g4 Rh4 Bg5 Rhh8 Bxc4 Rxc4 Rxc4 bxc4 Kf2 Nc6 h4 Rb8 Ra1 Rb2 Ke1 a5 Ra4 Rb1 Kf2 Rb2 Kg3 Rb3 Rxc4 Ra3 Kf4 a4 Ke4 Na5 Rc5 Nb3 Rb5 Kc6 Rb4 Ra1 Be3 Na5 c4 a3 d5 Kd7 Ra4 Nb3 Ra7 Ke8 d6 a2 d7 Ke7 Bg5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Be3 Be7 Nbd2 O-O c3 Bg4 Nxe4 dxe4 Qd5 exf3 Qxc6 fxg2 Qxg2 Qd7 Bh6 gxh6 f3 h5 Rad1 Qf5 fxg4 Qxe5 Rde1 Qc5 Kh1 Rad8 Rf5 Qd6 Rd5 Qg6 Rxe7 Rxd5 Bxd5 hxg4 Qe4 Qxe4 Bxe4 Rd8 Rxc7 h5 Bc2 Rd5 Bb3 Rf5 Kg2 a5 Rxf7 Rxf7 Kg3 a4 Bxf7 Kxf7 Kh4 Kg6 b3 a3 c4 bxc4 bxc4 Kf5 Kxh5 Ke4 Kxg4 Kd4 h4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 d6 Bc4 Be7 d3 Nf6 c3 O-O O-O c6 Bb3 Be6 Bc2 h6 Re1 Nbd7 Nbd2 Qc7 d4 Rfe8 h3 Nf8 c4 Ng6 d5 Bd7 Nb1 Bf8 Nc3 c5 Ba4 a6 Bxd7 Nxd7 g3 Be7 h4 Nf6 Nh2 Qd7 a4 Qh3 Qf3 Qd7 a5 Nf8 Bd2 Rec8 Nf1 Ng4 Na4 Bd8 Rec1 Rab8 b4 cxb4 Bxb4 h5 Nb6 Bxb6 axb6 Qe7 Qa3 Rd8 f3 Nh6 c5 dxc5 Bxc5 Qf6 Kg2 Re8 Be3 Nd7 Rab1 Qe7 Qxe7 Rxe7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 cxd5 cxd5 Nc3 Nc6 Bf4 Bf5 e3 e6 Bb5 Bd6 Ne5 Rc8 Bxc6 bxc6 O-O O-O Rc1 Qe7 Na4 Bxe5 Bxe5 Nd7 Nc5 f6 Bg3 e5 Qd2 Nb6 b3 e4 Na6 h5 Qb4 Qb7 Nc5 Qe7 a4 g5 a5 Na8 Na6 Qe8 Rc2 h4 Bd6 Rf7 Rfc1 Rg7 Nb8 Bd7 Qb7 Kh7 Qxa8 Qe6 Qxa7 Qxd6 Nxd7 Rxd7 Qb6 Rdc7 a6 Qd7 Ra1 h3 gxh3 Kg6 f3 exf3 Rf2 Ra7 Rxf3 Rca8 b4 Qc7 Qc5 Rxa6 Raf1 Ra2 Rxf6 Kh5 R1f2 Rxf2 Rxf2 Qh7 Qc2 Ra1 Kg2 Qe4 Qxe4 dxe4 d5 cxd5 Rb2 Ra6 b5 Rb6 Kg3 Kg6 Kg4 Kf6 Rb4 Kg6 h4 gxh4 Kxh4 Kf5 Kg3 Rg6 Kf2 Rh6 b6 Rxh2 Ke1 Rh8 b7 Rb8 Kd2 Ke5 Kc3 d4 exd4 Kd5 Kd2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 a6 Qd2 Qb6 Nb3 e6 Be2 Qc7 a4 b6 Bxf6 gxf6 f4 Bh6 O-O Bb7 Rf3 Rd8 Rh3 Bg7 f5 O-O Qf4 Ne5 Rd1 Qe7 Qh4 h6 Qf2 Qc7 Nd4 Bc8 Qf1 Qb7 Bc4 d5 exd5 exd5 Bb3 Rfe8 Rg3 Kh7 Qf2 Nc4 Nde2 b5 axb5 axb5 Nf4 Re5 Nh5 Bh8 Rf3 Nxb2 Rd4 Nc4 Bxc4 bxc4 g4 Qe7 Qd2 Bb7 Kf2 Re8 Nf4 Bg7 Rh3 Kg8 Nce2 Bf8 Nc3 Qc5 Nh5 Qb6 Rf3 Kh7 Kg3 Rd8 Kh4 Bb4 Qf4 Qd6 Rg3 Qe7 Kh3 Bc5 Qd2 Rd6 Nf4 Bxd4 Qxd4 Qd8 Nh5 Re8 g5 hxg5 Rxg5 Rg8 Rxg8 Kxg8 Nb5 Rc6 Qg4 Kf8 Nd4 Ra6 Ne6 Rxe6 fxe6 Ke7 Qg7 Kxe6 Nf4 Ke5 Kg3 Qa5 Kf2 Qb6 Kf3 d4 Kg3 d3 Qxf7 Qg1 Kh4 Qg5
Result: 0-1
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 Nc3 g6 g3 Bg7 Bg2 O-O Nge2 e5 h3 exd4 Nxd4 Re8 O-O Nbd7 a4 a5 Ndb5 Nc5 Bg5 Bd7 Qd2 h6 Bxf6 Bxf6 Nd5 Bxb5 axb5 Bxb2 Rxa5 Rxa5 Qxa5 c6 Qxd8 Rxd8 bxc6 bxc6 Ne7 Kg7 Nxc6 Rc8 Na7 Re8 Nb5 Nxe4 Rb1 Bf6 Bxe4 Rxe4 Nxd6 Re6 Rd1 Be7 Nb5 Rc6 Rd7 Kf6 Rd2 Bb4 Re2 Rc5 Nd4 Rc4 Nf3 Bc3 Kg2 Kg7 h4 f5 Re7 Kf8 Ra7 Bf6 Ne1 f4 Ra2 fxg3 fxg3 Be5 Nf3 Bd6 Ng1 h5 Ne2 Be7 Nf4 Bd6 Ne2 Be7 Kf3 g5 hxg5 Bxg5 Ra5 Bd2 Rxh5 Rxc2 g4 Kf7 Rd5 Bb4 Nf4 Rc3 Ke4 Rc4 Kf3 Rc3 Ke2 Rc2 Ke3 Ba3 Nd3 Kg6 Kf4 Rc4 Ke3 Rc2 Kd4 Bb2 Ke4 Rc4 Kf3 Rc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Be2 Bg7 Nf3 O-O O-O Bg4 Bf4 Re8 Nd2 Bxe2 Qxe2 Nh5 Be3 Nd7 a4 a6 a5 Qh4 Nc4 Nf4 Qf3 Ne5 Nxe5 Bxe5 g3 Qh5 Qxh5 Nxh5 Na4 Bg7 f3 f5 Nb6 Rab8 Nc4 fxe4 Nxd6 Red8 Nc4 Rxd5 fxe4 Rd7 e5 Bf8 Rad1 Rbd8 Rxd7 Rxd7 e6 Rc7 Bg5 Ng7 Bd8 Nxe6 Bxc7 Nxc7 Rd1
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 e4 c5 d5 exd5 exd5 d6 Nc3 Nf6 Be2 Be7 Nf3 O-O O-O Na6 h3 Nc7 Nh2 a6 a4 b6 Ng4 Bb7 Ne3 Re8 Rb1 Bf8 b4 cxb4 Rxb4 Ncxd5 Nexd5 Nxd5 Nxd5 Bxd5 Be3 Be4 Bxb6 Qg5 Rxe4 Rxe4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6 Bc4 Nb6 Bb5 dxe5 Nxe5 Bd7 Nc3 Nxe5 dxe5 Bxb5 Nxb5 Qxd1 Kxd1 Nd5 Ke2 a6 Rd1 O-O-O Na3 e6 Nc4 Be7 Bd2 b6 g3 Kb7 Ne3 Nc7 Nc4 Rd4 Rac1 Nd5 f3 Rc8 Ne3 Nxe3 Bxe3 Rxd1 Rxd1 Rc2 Rd2 Rxd2 Kxd2 Kc6 f4 b5 Kd3 Kd5 h4 h5 Bf2 Bb4 b3 g6 Ke2 Bc5 Bxc5 Kxc5 Kd3 Kb4 Kc2 Ka3 Kb1 a5 Ka1 a4 bxa4 Kxa4 Kb1 Ka3 Ka1 b4 Kb1 b3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 Be7 Be2 O-O O-O b6 Be3 Ne5 f4 Ned7 Bf3 Bb7 Qe2 Qc7 Rac1 Rac8 g4 Nc5 Qg2 d5 e5 Nfe4 cxd5 exd5 b4 Nxc3 Rxc3 d4 Bxd4 Qd7 Nc2 Bxf3 Rcxf3 Ne6 Be3 f5 exf6 Bxf6 Kh1 Qd5 a3 Qc4 f5 Qxc2 Qxc2 Rxc2 fxe6 Rc6 a4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Be7 O-O O-O f4 Nc6 Be3 a6 a4 Re8 Kh1 Qc7 Qe1 Nxd4 Bxd4 e5 fxe5 dxe5 Qg3 Bd8 Be3 Be6 Rad1 Kh8 Bg5 Ng8 Be3 Nf6 Bg5 Ng8 Be3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 d6 Nc3 Nbd7 e4 e5 Nf3 c6 a4 h6 Be3 Ng4 Bc1 exd4 Qxd4 Nde5 Be2 Be7 O-O Bf6 Qd1 Nxf3 Bxf3 Ne5 Be2 O-O f4 Ng6 Qd3 Qe7 Bd2 b6 Rae1 Bb7 Bh5 Rad8 b4 a6 Be3 b5 Bb6 Rde8 axb5 axb5 Bxg6 Bxc3 Bh7 Kxh7 e5 Kh8 Qxc3 c5 bxc5 dxe5 c6 Qe6 f5 Qf6 Bc7 Ba8 Ba5 Kh7 c7 Bb7 Qc5 g6 Qb6 Qxb6 Bxb6 gxf5 Rd1 Rg8 g3 Re6 Ba5 Ra8 Bb4 Rc6 Rd7 Rxc2 Rfd1 Kg6 R1d2 Rc4 Bd6 Ra1 Kf2 Rh1 Ke3 f4 gxf4 exf4 Bxf4 Re4 Kd3 Rxf4 Rg2 Kh7
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 g4 Nc6 g5 Nd7 Be3 Be7 Rg1 O-O Qh5 g6 Qe2 Nxd4 Bxd4 Bxg5 Be3 Bxe3 Qxe3 Qb6 O-O-O Qxe3 fxe3 Ne5 Rxd6 b6 Be2 Bb7 Rg5 f6 Rg1 Rad8 Rd4 Bc6 Rgd1 Rc8 a4 Kf7 Nb5 Bxb5 Bxb5 Rc7 b3 g5 Kb2 h5 Rd6 Ke7 a5 bxa5 Ra6 Rb8 Rxa5 Rd8 Rd4 g4 Be2 h4 Ra1 Rdc8 c4 Rg8 Kc3 g3 hxg3 Rxg3 Rh1 h3 Bf1 Rxe3 Kc2 Nf3 Rd1 Ng5 Rd3 Re1 Rd1 Rxd1 Kxd1 Nxe4 Ke1 Ng5 Bxh3 Rb7 Bg2 Rxb3 Rh8 Rb2 Bc6 Rc2 Bb5 Kd6 Kd1 Ra2 Rd8 Ke5 Bc6 Ne4 Bxe4 Kxe4 c5 Ra5 Rc8 Kd5 c6 Rc5 c7 Ke5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be3 a6 g4 Nc6 g5 Nd7 Rg1 Be7 h4 O-O h5 Nde5 Nxc6 Nxc6 f4 b5 Qf3 Bb7 Bd3 Nb4 f5 exf5 Qxf5 Nxd3 cxd3 Qc8 h6 Re8 hxg7 Qxf5 exf5 Bxg5 Rxg5 Rxe3 Kd2 Rf3 Ne4 Bxe4 dxe4 Re8 Rc1 d5 e5 h6 Rh5 Rxe5 f6 Rf2 Kd3 Rf3 Kd4 Re4 Kxd5 Re8 Rxh6 Rf5 Kd4 Rf4 Kc5 Re5 Kb6 Re6 Rc6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Rb1 Be6 b4 Qd7 b5 Nd8 Nd5 c6 bxc6 bxc6 Nc3 Ne7 Ba3 f5 Qb3 O-O Nf3 Nf7 O-O Qc8 Bb2 Rb8 Qa3 Qc7 Ba1 Rxb1 Rxb1 Rb8 Rxb8 Qxb8 Nd2 Bf6 Nb3 e4 Nd5 cxd5 Bxf6 dxc4 Bxe7 cxb3 axb3 exd3 exd3 Bxb3 h3 Qb6 Qc1 Be6 Bf6 Qc5 Qf4 Bd5 Qa4 Kf8 Qd7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 g3 b6 e4 Bb7 d3 d6 Bg2 Nbd7 O-O Be7 Re1 a6 b3 O-O d4 cxd4 Nxd4 Qc7 Bb2 Rfe8 Qd2 Bf8 Rad1 Rad8 h3 Qb8 Ba1 Qa8 Kh2 Nc5 Qc2 Rc8 Qb1 Red8 Re3 g6 f4 Bg7 Nde2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 f3 O-O Be3 Bd7 Qe1 Nc6 Nxc6 Bxc6 Ne2 Qb8 Nd4 Bd7 c3 a5 a3 e5 Nf5 Bxf5 exf5 Qc8 g4 Ne8 Bd5 Ra6 Qd2 Nc7 Rfd1 Nxd5 Qxd5 h5 h3 Kh7 a4 bxa4 Rxa4 g6 Rc4 Qe8 fxg6 fxg6 Rd2 h4 g5 Kg7 Rxh4 Rf5 f4 Qc8 Rg4 exf4 Bd4 Kf8 Qf3 d5 Rxf4 Bxg5 Rxf5 Qxf5 Qxf5 gxf5 Rf2 f4 Be5 Ke7 Bxf4 Bxf4 Rxf4 a4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 d6 e4 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 O-O Qe8 Nd2 Nh7 a3 f5 exf5 Bxf5 g4 Bd7 Nde4 a4 f3 b6 Bd3 Bf6 Nxf6 Nxf6 Qd2 Nc5 Bc2 Kg7 Rae1 Nb3 Qd3 g5 Bg3 Nc5 Qd2 Qf7 h4 Nh7 Bxh7 gxh4 Bxe5 dxe5 Bb1 Qf4 Qxf4 Rxf4 Rxe5 Raf8 Re7 R8f7 Rxf7 Kxf7 Ne4 Nb3 Kf2 Bxg4 Ke3 Rf5 Rf2 Bh5 Nd6 cxd6 Bxf5 Kf6 Bc2 Nc5 Rh2 Kg5 Bd1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Nd3 f5 Bd2 Nf6 f3 f4 c5 g5 Rc1 Ng6 cxd6 cxd6 Nb5 Rf7 Qc2 Ne8 a4 h5 Nf2 Bf8 h3 Rg7 Nxa7 Rc7 Ba5 Rxc2 Bxd8 Rxe2 Nxc8 Rxa4 Nd3 g4 Rf2 Re3 Ne1 g3 Rfc2 Rb3 Kf1 Ra2 Rb1 Kf7 Ke2 Ra8 Nd3 Be7 Bxe7 Nxe7 Nxe7 Kxe7 Rbc1 Kd7 Rc4 Rb6 Kd2 Rba6 Rb4 R8a7 Ne1 Nc7 Nc2 Ra4 Rxa4 Rxa4 Ra1 Rxa1 Nxa1 Na6 Nc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 cxd4 exd4 O-O a3 Be7 d5 exd5 cxd5 Re8 g3 Bc5 Bg2 d6 h3 Bf5 O-O Nbd7 g4 Be4 Ng3 Bxg2 Kxg2 Nf8 g5 N6d7 h4 Ne5 h5 f6 Nce4 fxg5 Bxg5 Qb6 h6 Nf7 hxg7 Nd7 Nf6 Nxf6 Bxf6 Qb5 Rh1 Bb6 Qf3 Ne5 Nf5 Nf7 Rxh7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 Nf3 g6 Bg5 Bg7 e3 h6 Bh4 g5 Bg3 Nh5 Nd2 Nxg3 hxg3 a6 g4 Nd7 a4 Ne5 Be2 O-O O-O f5 gxf5 Bxf5 Nde4 g4 Ng3 h5 Nxf5 Rxf5 Ne4 Rf7 f4 gxf3 Bxf3 Bh6 Qe2 Qh4 g3 Rg7 Rf2 Re8 Bg2 Rxg3 Nxg3 Ng4 Nf1 Bxe3 Nxe3 Qh2
Result: 0-1
Processing game...
FEN: 
Moves: c4 e6 Nf3 d5 g3 dxc4 Bg2 Nd7 Qa4 a6 Qxc4 b5 Qc6 Rb8 Qc2 Bb7 a4 Ngf6 O-O c5 d3 Be7 axb5 axb5 Bf4 Rc8 Nc3 Qb6 Ne5 Bxg2 Kxg2 Nxe5 Bxe5 O-O Qb3 Qc6 Kg1 b4 Bxf6 gxf6 Nd1 c4 dxc4 Qxc4 Qxc4 Rxc4 Ne3 Rc7 Rfc1 Rfc8 Rxc7 Rxc7 g4 Bd6 Rd1 Be5 b3 Kf8 Nc4 Ke7 Kg2 Bc3 f4 Rc5 Kf3 f5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 Be7 Be2 O-O O-O b6 Be3 Ne5 Qb3 Ned7 Rfd1 Qc7 f3 Bb7 Kh1 Rac8 Rd2 Nc5 Qd1 Rfd8 Rc1 Qb8 Qg1 d5 exd5 exd5 Rcd1 dxc4 Rxd8 Rxd8 Rxd8 Qxd8 Nxc4 Qc7 a4 Nd5 Nxd5 Bxd5 Qd1 Be6 Qd4 Bf6 Qd6 Qxd6 Nxd6 Nxa4 Bxa6 Be7 Ne4 f5 Nc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 c5 d5 d6 c4 e5 e4 Be7 Nf3 Bg4 Nbd2 Na6 Be2 Nf6 a3 O-O O-O g6 Ne1 Bd7 Nd3 Ne8 b4 Rc8 Bb2 Bf6 Qb3 b6 Kh1 Bg7 f4 Bh6 Rf2 f6 b5 Nb8 Bf1 Rc7 g3 Bc8 Bg2 Rcf7 a4 Ng7 Bc3 Nh5 Qd1 Ng7 a5 Ne8 Qe2 Bg7 axb6 axb6 f5 Bh6 h4 Rg7 Nf1 Rff7 Ra8 Bb7 Ra2 Bc8 Bh3 Ra7 Qf3 Rgb7 Nh2 Qc7 Kg2 Bg7 Bg4 gxf5 Bxf5 Bxf5 Qxf5 Qd7 h5 Qxf5 exf5 Bh6 Ng4 Bg5 h6 Re7 Rfe2 Kf7 Ndf2 Bc1 Ne4 Red7 Kf3 Ke7 Rxa7 Rxa7 Re1 Bg5 Nxg5 fxg5 Nxe5 dxe5 Bxe5 Nd6 f6 Kd7 Bxd6 Kxd6 Re6 Kc7 f7 Ra1 Ke2
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 d4 b6 g3 Bb7 Bg2 Be7 O-O O-O Nc3 Ne4 Nxe4 Bxe4 Ne1 Bxg2 Nxg2 d5 Qa4 dxc4 Qxc4 c5 dxc5 Bxc5 Be3 Nd7 Rfd1 Qe7 Bxc5 Nxc5 Rac1 Rfd8 Ne3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6 Qc2 Bb7 Bxf6 Bxf6 cxd5 exd5 O-O-O c5 dxc5 Nd7 Nxd5 Nxc5 Bc4 b5 Nxf6 Qxf6 Bd5 Rac8 Kb1 Na4 Qe2 Bxd5 Rxd5 Rc4 Rd4 Rfc8 Rhd1 Qg6 Qd3 Qxg2 Qf5 Rf8 Rd8 Rc7 Rxf8 Kxf8 Nd4 Re7 Nxb5 Rxe3 Nd6 Rf3 Qc8 Ke7 Qe8 Kf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 Nc6 g3 e6 Bg2 Bc5 Nb3 Be7 O-O d6 Nc3 O-O Bf4 Nh5 Be3 Ne5 c5 d5 Bd4 Nc6 e4 Nxd4 Qxd4 dxe4 Qxe4 Nf6 Qf4 Nd5 Nxd5 exd5 Rad1 Be6 Rd3 Rc8 Bxd5 Bxd5 Rfd1 Bxc5 Rxd5 Qb6 Rd7 Rc6 Nxc5 Rxc5 b4 Rc2 a4 Rb2 b5 a6 Qd4 Qxd4 R1xd4 axb5 Rxb7 b4 Rdxb4 Ra2 h4 h5 g4 Ra8 Rb8 Rxb8 Rxb8 Kh7 Rb4 hxg4 Rxg4 f5 Rb4 Kh6 Kg2 g5 hxg5 Kxg5 Kg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Be7 cxd5 exd5 g3 c6 Bg2 Bd6 O-O Re8 Qc2 Nbd7 b4 Nf8 f3 a5 Rb1 axb4 axb4 Ne6 e4 Qb6 Na4 Rxa4 Qxa4 dxe4 Kh1 exf3 Bxf3 Nxd4 Nxd4 Qxd4 Bb2 Qc4 Bxf6 gxf6 Bg2 Kg7 Qd1 Be5 b5 Bg4 Qd2 cxb5 Rb4 Qe2 Rf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 O-O Nc6 Be3 cxd4 Nxd4 Nxd4 Qxd4 Ne7 Bb5 Nc6 Qa4 Qc7 c4 dxc4 Rc1 Be7 Nd2 O-O Bxc6 bxc6 Nxc4 Be4 Nb6 axb6 Qxe4 c5 a4 Rad8 Rd1 Rd7 Rd2 Rxd2 Bxd2 Rd8 Bc3 Rd5 Qc4 h5 Qe2 g6 a5 bxa5 Rxa5 Bg5 g3 Bh6 Ra8 Kh7 Ra1 Bg7 Re1 c4 h4 Qc5 Qe3 Qc6 Qf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Qe7 d5 Ng4 Bg5 f6 Bh4 h5 h3 Nh6 Nd2 c5 Nf1 Nf7 g4 hxg4 Bxg4 g5 Bxc8 Rxc8 Ne3 gxh4 Nf5 Qd8 Qg4 Ng5 Nxh4 Rc7 Nf5 a6 h4 Nh7 Rg1 Qf8 Ke2 Ra7 a4 b6 Qh5 Kh8 Rg6 Rd7 Rag1 Rab7 Qg4 Rbc7 Rg2 Rb7 Kf1 Ra7 Kg1 Rf7 Ne2 Qc8 f4 b5 axb5 axb5 cxb5 Rab7 h5 Nf8 Qh3 Nxg6 hxg6 Kg8 gxf7 Kf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Bf4 Bg7 Qa4 Bd7 Qb3 Qc7 Nf3 O-O e5 Re8 Be2 Nh5 Be3 dxe5 O-O Qb6 Nd2 Nf4 Bc4 Na6 Rfc1 Bf5 Qd1 g5 Nb3 Qg6 Bxa6 bxa6 Nxc5 Rad8 Qa4 e4 Qxa6 Qh5 Qf1 Be5 Bxf4 Bxf4 g3 Bxc1 Rxc1 Rd6 Re1 Rh6 Qg2 Bh3 g4 Qxg4 Qxg4 Bxg4 N5xe4 Kf8 Rc1 Bf3 Nd2 g4 Nxf3 gxf3 Rd1 Re5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 exd5 g3 Nbd7 Bg2 a6 Bf4 Nh5 Bg5 Be7 Bxe7 Qxe7 Nh4 Nhf6 Nf5 Qf8 Qb3 O-O-O O-O Nb8 Rac1 Nc6 Nxd5 Nxd5 Bxd5 Nxd4 Qc4 Nxe2 Qxe2 Bxd5 Qxa6 Kb8 Rxc7 Kxc7 Qa7 Kc6 Rc1 Qc5 Rxc5 bxc5 b4 cxb4 axb4 Rhe8 Ne7 Kd6 Qc5 Ke6 Nxd5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d4 d6 c4 Nbd7 Nc3 e5 e4 c6 Re1 exd4 Nxd4 Re8 h3 a5 Be3 Nc5 Qc2 a4 Rab1 Qe7 Rbd1 Nfd7 b4 axb3 axb3 Na6 Na2 Ndc5 b4 Nxe4 Bxe4 Qxe4 Bd2 Qxe1 Bxe1 Bxd4 Rxd4 Rxe1 Kh2 Bf5 Qd2 Rae8 Nc3 Nxb4 Rxd6 h5 Rd8 Kh7 Rxe8 Rxe8 Qb2 c5 Nd5 Nxd5 cxd5 c4 Qxb7 Kg8 d6 c3 d7 Rd8 Qc7 Rxd7 Qxc3 Be6 f3 Rd5 g4 hxg4 fxg4 Rb5 Qd4 Rd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 d6 e4 O-O c3 Nbd7 Be2 e5 dxe5 dxe5 O-O b6 Re1 Bb7 Qc2 h6 Bh4 Qe7 Bf1 Rfe8 b4 a6 Nc4 Rac8 a4 Qe6 Nfd2 Nh5 f3 Bf6 Bf2 Bg5 Ne3 Ndf6 c4 c6 Nb3 Nd7 c5 b5 Red1 Be7 Nc4 Rc7 Nd6 Rb8 axb5 cxb5 Nxb7 Rbxb7 Qa2 Nb8 Na5 Qxa2 Rxa2 Ra7 c6 Ra8 Rc2 Bxb4 Rd8 Kg7 Bb6 Bxa5 Bxa5 Rxc6 Rxb8 Rxb8 Rxc6 b4 Bc7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 c5 d5 exd5 cxd5 d6 e4 g6 f4 Bg7 Bb5 Nfd7 a4 O-O Nf3 a6 Be2 Nf6 O-O Qc7 e5 Ne8 e6 fxe6 Bc4 Qe7 dxe6 Nc6 f5 Nc7 Bg5 Bf6 Ne4 Bxg5 Nfxg5 gxf5 Nxd6 Nd4 Qh5 Bxe6 Rae1 Rf6 Nxf5 Nxf5 Nxe6 Nxe6 Rxe6 Rxe6 Qxf5 Re8 Re1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be2 Be7 O-O O-O Be3 a6 a4 Nc6 f4 Bd7 Kh1 Nxd4 Bxd4 Bc6 Bf3 Nd7 Qe2 Re8 e5 Qc7 Bxc6 Qxc6 Ne4 d5 Ng5 Bxg5 fxg5 Rac8 c3 Qc4 Qh5 Re7 b4 b6 Rf3 a5 b5 Qe2 Qh4 Qe4 Qxe4 dxe4 Re3 Nc5 Ra3 Rd7 Kg1 Rd5 Kf1 Nd7 Rxe4 Rc4 Re3 Nc5 Ke2 Nxa4 Kd3 Nb2 Ke4 Rxb5 Re1 f6 gxf6 gxf6 Kf3 Nd3 Re3 Nxe5 Bxe5 Rxe5 Rxe5 fxe5 Rb3 Kf7 Rxb6 Rxc3 Ke4 Kf6 Rb7 Rc4 Kf3 h5 Rh7 a4 Rh6 Kg5 Rxe6 Kf5 Ra6 Rc3 Kf2 Ra3 Rh6 Kg5 Re6 Kf4 g3 Ke4 Rh6 Ra2 Kg1 a3 Rxh5 Rb2
Result: 0-1
Processing game...
FEN: 
Moves: e4 g6 d4 d6 Bg5 Bg7 c3 Nf6 Nd2 Na6 Ngf3 O-O Be2 c5 O-O Nc7 dxc5 dxc5 Qc2 Ne6 Bh4 Nf4 Bc4 Qd7 a4 N6h5 Rfe1 Qg4 Bg3 e5 Rad1 Qd7 Nf1 Qc7 Ne3 Be6 Bxe6 Nxe6 Nd5 Qb8 Bh4 Kh8 Nd2 b5 axb5 Qxb5 Ra1 Nhf4 Qa4 Qd3 Rad1 Nxd5 exd5 Qxd5 Nc4 Qb7 Rd7 Qb8 Be7 Re8 Red1 Bf8 Bf6 Bg7 Bxg7 Kxg7 Nd6 Rf8 Rb7 Qd8 Nxf7 Qf6 Rdd7 Qf4 Qxf4 exf4 Ng5 Kf6 Nxh7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 g3 O-O Bg2 d6 O-O e5 d3 Nc6 Rb1 a5 a3 Nd4 b4 axb4 axb4 c6 b5 Bg4 bxc6 bxc6 e3 Nxf3 Bxf3 Bxf3 Qxf3 d5 cxd5 cxd5 e4 d4 Nd1 Nd7 Bd2 Nc5 Rb5 Ne6 Nb2 Rb8 Rd5 Rxb2 Rxd8 Rxd8 Rd1 Ra8 h4 Raa2 Bg5 Bf8 Bf6 Bh6 Bxe5 Rxf2 Qxf2 Be3 Qxe3 dxe3 Re1 e2 d4 Rd2 Kf2 Nxd4 Bxd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 Bg5 Be6 Bxf6 Bxf6 Qd3 Nc6 Nd5 Bg5 O-O Ne7 Nxe7 Qxe7 Rfd1 Rd8 Na5 O-O Nc4 d5 exd5 Rxd5 Qf3 Rfd8 Rxd5 Bxd5 Qf5 b5 Ne3 Bb7 Rd1 Rxd1 Bxd1 g6 Qd3 Qb4 Qd7 Qe7 Qd3 e4 Qd2 Qd8 Qxd8 Bxd8 c3 Kg7 a4 bxa4 Bxa4 f5 g3 Kf6 h4 Bc8 Bd1 h6 Be2 a5 Bb5 Ke5 Nc4 Kd5 Ne3 Kc5 Be8 g5 hxg5 hxg5 Bg6 f4 gxf4 gxf4 Ng2 Kc4 Nxf4 Kb3 Bxe4 Kxb2 Bc6 Kxc3 Kf1 Kd2 Bb5 Bf5 Nd5 Bd3 Bxd3 Kxd3 Ke1 a4 Kd1 a3 Kc1 Kc4 Ne3 Kb3 f4 Bc7 Kb1 a2 Ka1 Ba5 Nd5 Bd2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 Nd7 e4 Nxc3 bxc3 c5 O-O cxd4 cxd4 O-O Qe2 Rc8 Bb2 Qc7 Qe3 Nf6 Ne5 b5 f4 Qb6 Kh1 b4 axb4 Bxb4 Rab1 a5 Qe2 Qa7 f5 Qa8 d5 exd5 Ng4 Nxg4 Qxg4 f6 Bxf6 Rxf6 e5 Rh6 f6 Rc7 e6 Qd8 e7 Rxe7 fxe7 Qxe7 Rbc1 Qd8 Qf5 Qb8 Qf7 Kh8 Rc7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 g6 b3 Bg7 Bb2 c5 c4 d6 Bg2 e5 O-O Nc6 Nc3 O-O d3 Ne8 Nd2 Nc7 e3 Be6 Rc1 Qd7 Re1 Rad8 Nde4 h6 f4 f5 Nf2 exf4 gxf4 Qf7 Qd2 g5 Ne2 d5 Bxg7 Qxg7 Kh1 dxc4 bxc4 Ne8 Qc3 Nf6 d4 Bc8 Rcd1 cxd4 exd4 Ne7 d5 Ng6 Nd3 Nh4 Rg1 b5 Ne5 bxc4 Qxc4 Bb7 d6 Kh7 Bxb7 Qxb7 Qc6 Qxc6 Nxc6 Ne4 Rgf1 Rxd6 Rxd6 Nxd6 fxg5 hxg5 Rd1 Re8 Ned4 Re3 Kg1 Ne4 Rb1 Rd3 a4 Rd2
Result: 0-1
Processing game...
FEN: 
Moves: c4 e6 g3 d5 Bg2 Nf6 Nf3 Be7 O-O O-O b3 c5 Bb2 Nc6 e3 d4 exd4 cxd4 Re1 Re8 a3 a5 d3 Bc5 Ne5 Nxe5 Rxe5 Qc7 Re1 e5 Nd2 Bg4 Qc2 Rab8 Ne4 Nxe4 Bxe4 Bd7 Bc1 Bc6 f4 exf4 Bxf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 c3 d5 exd5 exd5 d4 Bd6 Be3 c4 b3 cxb3 axb3 Ne7 c4 Nbc6 c5 Bc7 Nc3 O-O Bd3 Bf5 O-O Qd7 Ra4 a5 Ne1 b6 Nb5 Nb4 Nxc7 Qxc7 Bxf5 Nxf5 Nd3 b5 Ra1 Nc6 Qg4 Qd7 Rfe1 f6 Nf4 Rf7 Qe2 b4 Qd3 Re8 Ne2 Rfe7 Bd2 h5 f3 h4 Kf2 g5 Rad1 Kg7 Bc1 Nh6 Ng1 Qf5 Rxe7 Rxe7 Qxf5 Nxf5 Ne2 a4 bxa4 Ra7 g4 hxg3 hxg3 Rxa4 g4 Nh4 f4 Ra2 fxg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 Nc3 Be7 O-O O-O Bg5 c4 Ne5 Be6 f4 Nxe5 fxe5 Ne4 Bxe7 Nxc3 bxc3 Qxe7 e4 Qd7 a4 Rfd8 Qh5 Rac8 Rf4 Rc7 Raf1 Qxa4 exd5 Rxd5 Bxd5 Bxd5 e6 Bxe6 d5 Qb5 Rh4 Qc5 Rf2 Bxd5 Rd4 Rd7 Rf5
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nf3 Bg7 Nc3 d6 d4 Nf6 e4 O-O Be2 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5 Nbd7 Nd5 c6 Ne7 Kf8 Nxc8 Rdxc8 O-O-O Nc5 Bxf6 Bxf6 Bd3 a5 Rhe1 Re8 Bf1 Bd8 g3 a4 Kc2 Ba5 Re3 Rad8 Rxd8 Rxd8 Bh3 f6 Re2 Ke7 Bg2 Nd3 a3 Nc5 h4 h5 Re3 g5 hxg5 fxg5 Re2 Nb3 Kb1 Kf6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 O-O c3 d6 e4 c5 dxc5 dxc5 Be2 Nc6 O-O b6 Qc2 Bb7 Bh4 Nh5 Rfd1 Qc7 Nc4 Bf6 Ne3 e6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 O-O e4 Nxc3 bxc3 c5 O-O cxd4 cxd4 Nc6 Be3 Rc8 Ra2 Bf6 Bb1 g6 Qd3 Qd7 Rd2 Rfd8 Qe2 Na5 Ne5 Bxe5 dxe5 Qc7 Rxd8 Rxd8 Bg5 Rd4 Qe3 Qxe5 f4 Qd6 Be7 Qd7 Bf6 Rd1 Bc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Qf3 b5 O-O-O b4 e5 Bb7 Ncb5 axb5 Bxb5 Nfd7 Nxe6 fxe6 Qh3 Kf7 f5 Be4 fxe6 Kg8 Qb3 Bxc2 Qxc2 Qxc2 Kxc2 Nxe5 e7 Bxe7 Bxe7 Nbc6 Bxd6 Ra5 Rd5 Rxb5 Rxb5 Nd4 Kb1 Nxb5 Bxe5 Kf7 Rc1 Ke6 Rc5 Nd6 Kc2 Rf8 Bd4 Rf1 Re5 Kd7 Bc5 Kc6 Bxb4 Rf2 Kc1 Nb7 Rg5 g6 a3 Re2 Kb1 Rf2 Ka2 Nd8 Rc5 Kb7 Rd5 Nc6 Bc3 Rxg2 Rd7 Kb6 Rxh7 Kb5 Kb3 Rg4 Rd7 Rh4 Rd2 Rh3 Rg2 Nd4 Ka2 Nf5 Be5 Re3 Bb8 Re6 Rg4 Ne7 Bg3 Nc6 a4 Kc5 Ka3 Rf6 b3 Kd5 Rg5 Ke4 Rc5 Re6 a5 Kd3 b4 Nd4 Be5 Nf3 Bb2 Nd2 Rc3 Ke2 Ka4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O O-O d6 Nc3 e5 d4 Nbd7 h3 c6 e4 Qb6 dxe5 Nxe5 Nxe5 dxe5 Qc2 Nh5 Be3 Qc7 Qe2 Be6 Rfd1 Rfd8 Na4 Rxd1 Rxd1 Rd8 b3 Rxd1 Qxd1 Bf8 Qd2 Nf6 Nc5 Bc8 Qc3 Nd7 Nd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c6 Nf3 g6 g3 Bg7 Bg2 Nf6 O-O O-O Nc3 d5 cxd5 cxd5 d3 Nc6 Qb3 d4 Nb5 a6 Na3 b5 Bf4 Be6 Qc2 Rc8 Qd2 Bd5 Rfc1 Nd7 Nc2 e5 Bh6 Qe7 Bxg7 Kxg7 e3 dxe3 Nxe3 Bxf3 Bxf3 Nd4 Bg2 Nc5 Re1 h5 Rac1 Rfd8 Rc3 Qf6 Nc2 b4 Nxb4 e4 Qe3 a5 Nc2 Nxd3 Rxc8 Rxc8 Qxd4 Qxd4 Nxd4 Nxe1
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d3 d5 Nd2 c5 Ngf3 Nc6 g3 Nf6 Bg2 Be7 O-O b6 Re1 Bb7 e5 Nd7 h4 Qc7 Qe2 h6 Nf1 O-O-O N1h2 Rdg8 Ng4 Nf8 Bf4 g5 hxg5 hxg5 Bd2 Rh5 c3 Ng6 b4 c4 d4 Rgh8 a4 a6 Qd1 Nb8 b5 Nd7 Qc2 a5 Kf1 Qd8 Ke2 Qg8 Kd1 Nf4 Bf1 Rh1 Kc1 Rxf1 Rxf1 Nd3 Kb1 Qg6 Ne1 Qe4 Nxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 d6 Nc3 g6 e4 Bg7 Bd3 O-O Nge2 e6 O-O exd5 cxd5 b6 Bg5 h6 Bf4 Ba6 Ng3 Bxd3 Qxd3 a6 Rae1 Ng4 h3 Ne5 Bxe5 dxe5 a4 Nc6 Rb1 Qd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 Nc3 Ne4 Bd2 Bf6 O-O O-O Rc1 c5 d5 exd5 cxd5 Nxd2 Nxd2 d6 Nde4 Re8 Qd2 a6 b4 Be7 bxc5 bxc5 Qf4 Qc7 Na4 Qa5 Rb1 Bxd5 Nb6 Bxe4 Bxe4 Ra7 Nc8 Nc6 Nxa7 Nxa7 Bd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O d5 exd5 Nh4 c6 cxd5 Nxd5 Nf5 Nc7 Nc3 d5 e4 Bf6 exd5 cxd5 Bf4 Nba6 Re1 Qd7 Bh3 Kh8 Ne4 Bxb2 Ng5 Qc6 Ne7 Qf6 Nxh7 Qd4 Qh5 g6 Qh4 Bxa1 Nf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2 Nc6 d5 Bxc3 Bd2 Bxd2 Qxd2 Nd4 Nxd4 cxd4 Qxd4 Qa5 Qd2 Qxd2 Kxd2 Rd8 Ke3 b6 Rbc1 e6 Bc4 e5 Bb3 Bd7 Rc7 a5 d6 b5 f4 exf4 Kxf4 Ra6 e5 a4 Bd5 a3 Rf1 Ra4 Ke3 Be6 Bxe6 fxe6 Rff7 Rh4 Rg7 Kh8 Rge7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 e4 b5 e5 Nd5 a4 e6 axb5 Nxc3 bxc3 cxb5 Ng5 Bb7 Qh5 Qd7 Be2 h6 Bf3 Nc6 O-O Nd8 Ne4 a5 Bg5 Bd5 Rfe1 Nc6 Bh4 Ra7 Qg4 Rh7 Nd6 Bxd6 Bxd5 Be7 Be4 g6 Bf6 Kf8 Qf3 Nd8 d5 exd5 Bxd5 Qf5 Qe3 Rd7 Rad1 Bxf6 exf6 Ne6 Be4 Rxd1 Bxf5 Rxe1 Qxe1 gxf5 Qe5 Kg8 Qg3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 Nbd2 Bb4 Qb3 Nc6 d5 Bxd2 Bxd2 Ne7 Bc3 Nf5 Nd2 Nd6 f3 O-O e4 exd5 cxd5 Bxf1 Rxf1 a5 e5 a4 Qc2 Qe8 Kf2 Nxd5 Qd3 Qe6 exd6 c6 Qd4 Qh6 h4 f5 f4 Rf6 Rae1 b5 Nf3 Rxd6 Qc5 Rf8 Re2 Re6 Rxe6 Qxe6 Re1 Qh6 Bd4 Qg6 a3 Ra8 h5 Qxh5 Qd6 Qf7 Ng5 Qh5 Qxd7 Qh2 Kf3 Qh5 Kg2 Nxf4 Kg1
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 e3 Nc6 d4 d5 a3 cxd4 exd4 Be7 Bg5 O-O Bd3 dxc4 Bxc4 Nd5 Bxe7 Ncxe7 Nxd5 Nxd5 O-O b6 Qa4 Nf4 Qc6 Rb8 Qe4 Qf6 Bd3 Nxd3 Qxd3 Bb7 Ne5 Rfd8 Rad1 Qg5 Qg3 Qxg3 hxg3 Rbc8 Rd2 f6 Nf3 Rc4 Rfd1 h5 Ne1 Be4 f3 Bf5 Kf2 Rd5 Ke3 g5 Ra1 Rb5 b4 Rd5 Raa2 Rc3 Ke2 Bb1 Rab2 Bg6 Nc2 Rd7 Ra2 Rc4 b5 Kg7 Ke3 Rd5 Rb2 Be8 Kf2 Rxb5 Rxb5 Bxb5 d5 exd5 Ne3 Rc5 Nxd5 Bc4 Nc3 Be6 Ne4 Ra5 Rd3 Bf5 Ke3 Ra4 Rc3 Bxe4 fxe4 Kf7 Rd3 b5 Kf3 Ke6 Ke3 a6 Rc3 Rc4 Rd3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be3 Nf6 Bc4 a6 Qe2 Qc7 Bb3 Na5 g4 b5 g5 Nd7 f4 b4 Na4 Bb7 f5 e5 Nf3 Nxb3 cxb3 Bxe4 O-O Qb7 Nd2 Bc6 Rf2 Qb5 Qxb5 axb5 Rc1 Rc8 Nb6 Nxb6 Bxb6 Kd7 f6 Bd5 Rxc8 Kxc8 Ba5 gxf6 gxf6 Rg8 Kf1 Rg4 Ke2 Kd7 Nf1 Re4 Kd1 Be6 Ng3 Rd4 Kc2 d5 Nf5 Bxf5 Rxf5 Ke6 Rh5 h6 a4 bxa3 bxa3 Kxf6 Bc3 Re4 a4 bxa4 bxa4 Rxa4 Bxe5 Kg6 Rh3 Ra2 Kd1 Kf5 Bg3 Ke4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 g6 b3 Bg7 Bb2 c5 c4 d6 Bg2 e5 O-O O-O e3 Nc6 Nc3 Bf5 d4 e4 Ng5 Re8 dxc5 dxc5 Nb5 Re7 Qxd8 Rxd8 Rad1 Rxd1 Rxd1 h6 Nh3 g5 Nd6 Bg4 Rd2 Ne8 Bxg7 Kxg7 Kf1 Nxd6 Rxd6 Nb4 a3 Nd3 Ng1 Nc1 h3 Bc8 Ne2 Nxb3 Nc3 Be6 Bxe4 Na5 Bd5 Nxc4 Bxc4 Bxc4 Ke1 Be6 e4 Bxh3 f4 gxf4 gxf4 Rd7 e5 Rxd6 exd6 f6 Nd5 Kf7 Nc7 Bd7
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 c5 g3 d5 Bg2 Nc6 O-O d4 e3 e5 exd4 exd4 d3 Be7 Bf4 O-O Ne5 Nxe5 Bxe5 Ng4 Bf4 Bd6 Bxd6 Qxd6 Nd2 Qb6 Re1 Bd7 h3 Nf6 Qb3 Qa5 Nf3 Bc6 Ne5 Bxg2 Kxg2 Qc7 Re2 Rae8 Rae1 Re6 Nf3 Qc6 Rxe6 fxe6 Qb5 Qxb5 cxb5 Re8 Rc1 b6 Ne5 Nd5 Rc4 Ra8 a3 a5 bxa6 Rxa6 b4 Rxa3 bxc5 b5 Rxd4 Rc3 c6 Kf8 Rh4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 g3 b6 Bg2 Bb7 O-O e6 d4 cxd4 Qxd4 Be7 Nc3 d6 Rd1 a6 b3 Nbd7 Bb2 O-O h3 Qc7 e4 Rad8 Rac1 Rfe8 Nh2 Bf8 Nf1 Qb8 Kh1 Nc5 Re1 Qa8 b4 Ncd7 a3 Rc8 Ne3 Red8 Kh2 Qb8 Kh1 Nc5 Rc2 e5 Qd1 Ncxe4 Nxe4 Bxe4 Bxe4 Nxe4 Qf3 Nf6 Rd1 Qa7 g4 b5 g5 Ne8 Rdc1 bxc4 Rxc4 Rxc4 Rxc4 Nc7 Qc6 Ne6 h4 d5 Rc2 Nd4 Qc7 Qa8 Bxd4 exd4 Nf5 d3 Rd2 d4 Kh2 Re8 Qg3 Re1 Qxd3 Qh1 Kg3 g6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 g6 d4 Nf6 Bg5 Bg7 Nbd2 c5 Bxf6 Bxf6 Ne4 Bxd4 Nxd4 cxd4 Qxd4 O-O c4 Nc6 Qd2 d6 Nc3 Be6 e4 Qb6 Rd1 Ne5 b3 f5 Be2 f4 Nd5 Bxd5 Qxd5 Kg7 O-O Rac8 b4 Qxb4 Rb1 Qa3 Rxb7 Kf6 h4 h6 Rd1 Rb8 Rc7 Rfc8 Rxc8 Rxc8 Qb7 Qc5 Qb2 Qb6 Qc1 g5 Rd5 e6 hxg5 hxg5 Rd1 Ke7 Qc2 Rb8 Qa4 g4 Qa3 Qc5 Qc3 g3 Rf1 gxf2 Rxf2 Rb1 Bf1 Qe3 Qxe3 fxe3 Rc2 Nxc4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 b6 Bb5 c6 Bc4 O-O O-O Ba6 Bxa6 Nxa6 Qa4 Qc8 Bg5 Qb7 Rfe1 e6 Rab1 c5 d5 Bxc3 Red1 exd5 exd5 Bg7 d6 f6 d7 fxg5 Qc4 Kh8 Nxg5 Bf6 Ne6 Nc7 Nxf8 Rxf8 Rd6 Be7 d8=Q Bxd8 Qc3 Kg8 Rd7 Bf6 Qc4 Kh8 Qf4 Qa6 Qh6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6 Bd2 Qc7 O-O-O Ngf6 Ne4 O-O-O g3 Nxe4 Qxe4 Be7 Kb1 Rhe8 Qe2 Bd6 Rhe1 Re7 c4 c5 Bc3 Nf6 Ne5 cxd4 Rxd4 Bxe5 Rxd8 Qxd8 Bxe5 Rd7 Bc3 Qb6 g4 Qd6 f3 a6 a4 Qd3 Kc1 Kc7 Qxd3 Rxd3 Rf1 Kc6 Kc2 Rd7 a5 Ne8 Re1 Rd6 f4 Nf6 Bxf6 gxf6 Rd1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 b5 cxb5 a6 a4 h6 Bd2 e6 dxe6 Bxe6 Nf3 axb5 Bxb5 Na6 O-O Nc7 Re1 Nxb5 Nxb5 d5 exd5 Nxd5 Ne5 Re8 Rc1 Bf5 Nc6 Qd7 Rxc5 Rxe1 Qxe1 Re8 Qc1 Nb6 b3 Re2 Ba5 Be4 Ne5 Qe7 Nd4 Ra2 Bxb6 Bxe5 Qe3 Qxc5
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Qc2 g6 Bf4 Bg7 e3 O-O Nc3 Na6 Be2 c5 O-O cxd4 exd4 dxc4 Bxc4 Bg4 Rad1 Nh5 Bg5 Bxf3 gxf3 Rc8 Bb3 Bf6 Be3 Nb4 Qe4 Nc6 Rfe1 Na5 Bh6 Ng7 Bc2 a6 Bb1 Re8 Nd5 Nc6 a3 Nh5 Ba2 e6 Nxf6 Nxf6 Qf4 Nh5 Qg4 Nf6 Qf4 Nh5 Qd2 Ne7 d5 Nxd5 Bxd5 Qxd5 Qxd5 exd5 Rxe8 Rxe8 Rxd5 Ng7 Rd7 Nf5 Bg5 h6 Bf6 g5 Bc3 b5 Kf1 Re6 Rd8 Kh7 Rh8 Kg6 Rg8 Kh5 Rf8 Nd6 Bb4 Kg6 Bxd6 Rxd6 Rb8 Kf5 a4 bxa4 Rb4 Rd1 Kg2 Ra1 Rb6 a3 bxa3 Rxa3 Rxh6 Ra4 Rc6 a5 Kg3 f6 Rc5 Kg6 f4 gxf4 Kf3 Ra2 h4 a4 Ra5 a3 Ra8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Nf3 O-O Bd3 d5 O-O dxc4 Bxc4 Nbd7 Qe2 a6 a3 Ba5 Ba2 cxd4 exd4 Bxc3 bxc3 b5 a4 Bb7 Bd2 Ne4 Rfc1 Nd6 Bf4 Nf6 Nd2 Nf5 axb5 Nd5 Bg3 axb5 Qxb5 Nxc3 Rxc3 Nxd4 Qd3 Ra4 Nc4 Qd5 f3 Rfa8 Rd1 Rd8 Bb1 Nxf3 gxf3 Qc5 Ne3 Qxe3 Qxe3 Rxd1 Kg2 Rxb1 Rb3 Rb4 Rxb4 Rxb4 Qd2 Bxf3 Kxf3 Rb3 Kg2 h6 Qc2 Rb5 Kf3 Rf5 Kg2 h5 h4 g5 hxg5 Rxg5 Kh3 Rf5 Qc8 Kh7 Qd7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 e5 Nb5 d5 cxd5 Bc5 N5c3 O-O e3 e4 Be2 Qe7 Nd2 Rd8 a3 Nxd5 Nxd5 Rxd5 Qc2 Bf5 b4 Bb6 Bb2 Nc6 O-O Qg5 Kh1 Rd6 Nxe4 Bxe4 Qxe4 Rd2 b5 Rxe2 bxc6 Rxb2 cxb7 Rf8 Rac1 Ba5 Rc8 Qb5 Rfc1 Qxb7 Qe8 Qxc8 Qxc8 Bd2 h3 h6 Qc4 Bxc1 Qxc1 Rxf2 Qc7 a6 Qa7 Rf6 a4 Rd8 a5 Rd1 Kh2 Rd2 Qb8 Kh7 Qb4 Rff2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be2 Nf6 O-O Be7 Be3 O-O f4 e5 Nb3 exf4 Bxf4 Be6 Kh1 d5 e5 Nd7 Nxd5 Ndxe5 c4 Bg5 Qc1 Bxf4 Nxf4 Qe7 Qc3 Rad8 Rae1 Kh8 a3 f6 Rd1 Bc8 Nd5 Qf7 Nc5 b6 Ne4 Qg6 Nd6 Bg4 Bxg4 Qxg4 Nf4 Ng6 h3 Qd7 Nxg6 hxg6 b4 Ne5 Rd4 g5 Rfd1 Qc6 Qc2 Rd7 Qf5 Re7 Kg1 a5 Qc2 axb4 axb4 Rd7 Rd5 Rfd8 c5 bxc5 bxc5 Ra8 Qe4 Rc7 Nc4 Nf7 h4 Re8 Qf3 Qe6 Nd6 g4 Qf2 Ra8 Nxf7 Rxf7 Rd6 Qe5 R6d5 Qe6 Qf4 Re7 Kh2 Ra3 R5d3 Ra2 Rg3 f5 Rd6 Qe5 Qc4 Re2 Rd8 Re8 Rxe8 Qxe8 c6 Re4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Bxf6 gxf6 Be2 Nc6 Nb3 b5 Bh5 Bg7 Qg4 O-O O-O-O Ne7 Nd4 b4 Nce2 f5 exf5 e5 fxe5 dxe5 Qg5 h6 Qg3 Nxf5 Nxf5 Bxf5 Rd2 Rac8 Qb3 e4 Kb1 Be6 Qg3 Qa5
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Bb7 g3 c5 d5 Qb6 Bg2 O-O-O O-O b4 Na4 Qb5 a3 Nb8 axb4 cxb4 Be3 Bxd5 Bxd5 Rxd5 Qe2 Nc6 Rfc1 Na5 b3 c3 Nxc3 bxc3 Rxc3 Kd7 Qc2 Bd6 Rc1 Qb7 b4 Qxb4 Rb1 Qg4 Bxa7 e5 Qa2 Rd1 Rxd1 Qxd1 Kg2 Qh5 Qa4 Ke6 h4 Qe2 Qxa5 Ra8 Qa4 Kxf6 Qd7 Kg7 Rf3 Qc4 Qxd6 Rxa7 Qxe5 Kh7 Rf5 Qc6 Kh2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 h6 Bxf6 Qxf6 a3 dxc4 Ne5 c5 Nxc4 cxd4 Nb5 Qd8 Qxd4 Qxd4 Nxd4 Bd7 g3 Bc5 Nb3 Be7 Nca5 Bc6 Nxc6 Nxc6 Bg2 Rc8 Rc1 Bf6 Rc2 O-O Kd2 Rfd8 Kc1 Nd4 Nxd4 Bxd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qb3 a5 Na2 Be7 Nh4 Be4 Nc3 Nb6 Be2 O-O Nxe4 Nxe4 Nf3 Nd5 Ne5 Nb4 Bf3 Ng5 Bg4 Qd5 Bd1 Rfd8 Bd2 Ne4 Bc3 c5 Qxd5 Nxd5 Bf3 cxd4 Bxd4 Nd2 Rfd1 Nb3 Rab1 Bf6 Nc4 Bxd4 exd4 Nb4 Bxb7 Rab8 Bf3 g6 Kf1 Rbc8 Be2 Nc2 Nd2 Ncxd4 Nxb3 Nxb3 Rxd8 Rxd8 Rd1 Rc8 Ke1 Kg7 Rd3 Rc1 Bd1 Nc5 Rd8 Rb1 b3 Nxb3 Rc8 Rc1 Rxc1 Nxc1 Kd2 Na2 Bb3 Nb4 Kc3 Kf6 Kc4 Ke5 Kb5 Kd6 Kxa5 Kc5 f4 Nd5 Ka6 Kb4 Bd1 Ne3 Bf3 Kxa4 Kb6 Kb4 Kc6 Kc4 Kd6 Nd5 Ke5 Kd3 Bxd5 exd5 Kxd5 Ke3 Ke5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 a5 g3 O-O Bg2 b6 O-O Ba6 Qc2 Nc6 Bg5 Be7 a3 h6 Bxf6 Bxf6 Rd1 Qe7 e3 Rae8 Nfd2 g5 Nc3 Bg7 Nb5 Qd8 f4 Ne7 Nf3 Nf5 Qf2 c6 Nc3 gxf4 gxf4 Bxc4 e4 Nd6 Ne5 f5 Nxc4 Nxc4 b3 Nd6 e5 Nc8 Bf3 Kh7 Bh5 Re7 Kh1 Rg8 Rg1 Bh8 Ne4 fxe4 f5 Rg5 Rxg5 hxg5 f6 Kh6 fxe7 Qxe7 Bf7 d6 Rf1 g4 Bxe6 Qxe6 Qh4 Kg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 O-O Be7 dxc5 Bxc5 Bg5 O-O Nc3 d4 Bxf6 Qxf6 Nd5 Qd8 Nd2 a6 Rc1 Ba7 Nc4 Rb8 Nf4 b5 Nd6 Qxd6 Rxc6 Qd8 Qc2 a5 Rc1 Re8 Bd5 Bb6 Qb3 Re7 Bf3 Re5 Bh5 g6 Bxg6 hxg6 Rxg6 Kf8 Rh6 Ke7 Rcc6 Rf5 Qf3 Bc7 Qe4 Re5 Ng6 fxg6 Rh7 Kf8 Qxg6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 Nc3 c6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Bb7 g3 c5 d5 Qb6 Bg2 O-O-O O-O b4 Na4 Qb5 a3 Nb8 axb4 cxb4 Be3 Bxd5 Bxd5 Rxd5 Qe2 Nc6 Rfc1 Na5 b3 c3 Nxc3 bxc3 Rxc3 Kd7 Qc2 Bd6 Rc1 Qb7 b4 Qxb4 Rb1 Qg4 Bxa7 Be5 Rc5 Rxc5 Bxc5 Nc6 Qd3 Kc8 Rd1 Nb8 Rc1 Qa4 Bd6 Nc6 Bxe5 Rd8 Qb1 Rd5 Qb8 Kd7 Qc7 Ke8 Qxc6 Qxc6 Rxc6 Rxe5 Rc8
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qb3 a5 Na2 Be7 Nh4 Bg6 g3 Qb6 Nc3 Qxb3 Bxb3 Bd3 Rd1 Be4 Nxe4 Nxe4 f3 Nd6 e4 g5 Ng2 h5 Ne3 c5 dxc5 Nxc5 Bc2 Nd7 b3 f6 Kg2 Ne5 Bb2 Kf7 Rd2 Rad8 Rad1 b6 Bd4 Nc8 f4 gxf4 gxf4 Rhg8 Kf2 Ng4 Nxg4 Rxg4 Kf3 Rdg8 Bf2 e5 fxe5 fxe5 Bg3 Rf8 Kg2 h4 Rf1 Ke8 Rxf8 Kxf8 Kf3 Rg5 Bxh4 Rh5 Bg3 Na7 Rf2 Kg7 h4 Nc6 Kg4 Rh8 Bd3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qb3 a5 Na2 Be7 Nh4 Bg6 g3 Qc8 Nc3 O-O Nxg6 hxg6 Rd1 e5 Bf1 Bb4 Bg2 Re8 Na2 Bf8 Bd2 g5 Rac1 g4 Nc3 Qb8 Qc2 Qa7 d5 cxd5 Nxd5 Nxd5 Bxd5 Nf6 Ba2 b6 Qf5 Qb7 Bc3 Qf3 Qxf3 gxf3 g4 Nxg4 Rd7 Nh6 Bd5 Rac8 Bxf3 Bb4 Bd5 Red8 Rxd8 Rxd8 e4 Rc8 f3 Bd6 Kf2 Kf8 Ke2 f6 h4 Nf7 Kd3 Ke8 Rg1 Kf8 Rb1 Nh8 h5 Nf7 b4 axb4 Bxb4 Rd8 Bxd6 Rxd6 Kc4 Ng5 Rb3 Ke7 Kb5 f5 Ka6 f4 Kb7 Rh6 Rxb6 Rxb6 Kxb6 Nxf3 a5 Nd2 a6 f3 a7 f2 a8=Q f1=Q Kc7 Qc1 Bc6 Nxe4 Qe8 Kf6 Qg6 Ke7 Qxg7 Ke6 Qg4 Kf7 Qg6 Kf8 h6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 e3 d5 Nc3 Nc6 cxd5 exd5 Bb5 a6 Bxc6 bxc6 O-O Bd6 Qa4 Qc7 e4 cxd4 Nxd5 Nxd5 exd5 O-O Qxc6 Bb7 Qxc7 Bxc7 Nxd4 Bxd5 Bd2 Be5 Bc3 Bxd4 Bxd4 Rfc8 Bc3 Be6 Rfe1 Rc5 Re4 Rg5 f3 h6 Kf2 Rc5 a4 Rc4 Re5 Rd8 a5 Rd6 Rae1 Rcc6 R5e4 Rc4 R4e3 Rc8 g4 Rcd8 f4 Bxg4 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 c5 d5 exd5 cxd5 d6 e4 g6 f3 Bg7 Bg5 a6 a4 Nbd7 Nh3 h6 Be3 Ne5 Nf2 Bd7 Be2 g5 Qd2 Qe7 a5 Rb8 Na4 Nh5 Nb6 Bb5 O-O O-O b4 c4 Rac1 f5 Nxc4 Bxc4 Bxc4 g4 fxg4 fxg4 Nh1 Qh4 Be2 g3 Nxg3 Nxg3 hxg3 Qxe4 Rxf8 Rxf8 Bf4 h5 Rc7 Qb1 Qc1 Qg6 Qc2 Qe8 Qe4 h4 gxh4 Qd8 Rxb7 Qc8 Re7 Qd8 Bg5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 g3 Nf6 Bg2 Bd7 O-O Be7 a4 O-O Nce2 Rc8 c3 a6 h3 Qc7 g4 d5 exd5 Nxd5 Bxd5 exd5 Nf4 Bc5 Nb3 Ba7 Qxd5 Be6 Nxe6 fxe6 Qe4 Rxf2 Rxf2 Qg3 Qg2 Bxf2 Kf1 Qe5 Bf4 Qxf4 Qxf2 Qc4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Qe7 d5 Ng4 Bg5 f6 Bh4 h5 Nd2 c5 dxc6 bxc6 b4 Be6 O-O Nd7 Nb3 Nxh2 Kxh2 g5 Na5 Nb8 Bg3 h4 Bg4 Bxg4 Qxg4 hxg3 fxg3 a6 Nd1 d5 Ne3 Ra7 cxd5 Qxb4 Qe6 Raf7 Nac4 Qe7 Nf5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 e3 Nc6 d4 d5 cxd5 exd5 Bb5 Bd6 dxc5 Bxc5 O-O O-O b3 Bg4 Bb2 Rc8 Rc1 Bd6 Be2 Bb8 Nb5 Ne4 Nbd4 Re8 h3 Bxf3 Nxf3 Qd6 Qd3 Ng5 Rfd1 Rcd8 Kf1 Ne4 a3 a6 Qc2 Ba7 Bd3 Qe7 Re1 Rd6 b4 Re6 b5 axb5 Bxb5 h6 Rcd1 Rd8 Qb3 Qd6 a4 Bc5 Re2 b6 Kg1 Ne7 Nd4 Rg6 Bd3 Qd7 Kh1 Nf5 Bxe4 dxe4 Red2 Nh4 Ne6 Qxd2 Rxd2 Rxd2 Nf4 Rg5 Kg1 Nf3 Kf1 Bxe3 fxe3 Rdxg2 Qc3 Rh2 Ne2 Kh7 Qc8 Rh1 Kf2 Nd2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d4 d5 c4 dxc4 Na3 c3 bxc3 c5 Nc4 Nc6 Nce5 Bf5 Bb2 Be4 e3 Qc7 Qe2 Nxe5 Nxe5 Bxg2 Kxg2 Ne4 f3 Nd6 Rac1 Rfd8 Rfd1 Qa5 Ba1 Rac8 c4 cxd4 exd4 e6 Kh1 Bxe5 dxe5 Nxc4 Rxd8 Qxd8 Rxc4 Rxc4 Qxc4 Qd1 Kg2 Qxa1 Qc8 Kg7 Qxb7 Qxa2 Kh3 Qd5 Qxa7 Qxf3 Qa1 g5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 Nf3 Be7 Bg5 O-O e3 h6 Bxf6 Bxf6 Qd2 c6 h4 Nd7 g4 Re8 O-O-O b5 cxb5 cxb5 Bxb5 Rb8 g5 hxg5 hxg5 Bxg5 Nxg5 Qxg5 f4 Qf6 Qh2 g6 f5 Qg7 fxe6 fxe6 Rdg1 Rf8 Bd3 Rf6 e4 e5 Nxd5 Rf7 Rxg6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 c5 d5 Ba6 Qc2 exd5 cxd5 g6 Nc3 Bg7 g3 O-O Bg2 d6 O-O Re8 Re1 Qc7 Bf4 Nh5 Bd2 Nd7 Qa4 Bb7 Qh4 a6 Rac1 b5 b4 Qd8 Bg5 f6 Bd2 f5 Bg5 Qb6 e4 cxb4 axb4 Rac8 Be3 Qd8 Bg5 Qb6 exf5 Rxe1 Rxe1 Bxc3 Re7 Rc4 Qh3 Bc8 fxg6 Ndf6 Bxf6 Nxf6 gxh7 Kf8 h8=Q Kxe7 Qg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 Nxc3 bxc3 c5 O-O Nc6 e4 Rc8 Be3 cxd4 cxd4 O-O Qe2 Na5 Rfe1 Kh8 h4 Bxh4 Rad1 Be7 d5 Bc5 Bf4 f6 dxe6 Qe7 Bb1 Qxe6 Nh4 g5 Ba2 Qxe4 Qxe4 Bxe4 Rxe4 gxf4 Rxf4 Rcd8 Rxd8 Rxd8 Nf5 Nc6 Rg4 Ne5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Qa5 Qd2 O-O Rc1 Bg4 Be2 e6 d5 exd5 exd5 Bxf3 Bxf3 c4 O-O Nd7 d6 Ne5 Bxb7 Rad8 Bg5 Rd7 Rb1 f6 Bf4 Rfd8 Rfd1 Bf8 Bd5 Kh8 Bxe5 fxe5 Qg5 e4 Qh4 Qxc3 Qxe4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 b3 O-O Bb2 d6 e3 Nbd7 d4 a6 Rc1 b5 d5 exd5 Nxd5 Nxd5 cxd5 Bf6 Bxf6 Nxf6 Nh4 a5 e4 Re8 Re1 g6 Qd2 b4 a3 bxa3 Ra1 Qe7 h3 Qe5 Rxa3 Qd4 Qxd4 cxd4 Ra4 d3 Rd1 Nd7 b4 axb4 Rxb4 Nc5 Nf3 Ra2 e5 dxe5 Rc1 Rc2 Rxc2 dxc2 Rc4 Nb3 Rxc2 Bxd5 Rc7 e4 Ne1 Nd4 Rd7 Re5 Nd3 exd3 Bxd5 Kf8 Bc4 d2 Rxd4 Re1 Kg2 d1=Q Rxd1 Rxd1 h4 Rc1 Bd5 Rc5 Bb3 Ke7 Ba2 h6 Bb3 Rc7 Ba2 Kd6 Kh3 Ke5 Kg4 Ra7 Bb3 Rb7 Bd1 Rb2 Kf3 Kd4 Be2 Rb3 Kg2 Kc3 Bf3 Rb5 Bc6 Rc5 Be8 Rc7 Kf1 Kd2 Ba4 Rc1 Kg2 Ke1 Bb3 Rc7 f4 Ke2 Bd5 Ke3 Ba8 f6 Kg1 Rg7 Kg2 g5 hxg5 hxg5 fxg5 Rxg5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 c5 d5 Ba6 Qc2 Qe7 Bg5 exd5 Nc3 Bxc4 e4 h6 Bxf6 Qxf6 exd5 Bxf1 Kxf1 d6 Re1 Be7 Ne4 Qg6 Qa4 Kf8 h4 Qf5 Qd1 Na6 Ng3 Qd7 Re4 Nc7 Rf4 Qb5 Kg1 Qxb2 Nf5 Bf6 Nxd6 Rd8 Nc4 Qa1 Qxa1 Bxa1 Nce5 Bxe5 Nxe5 Nxd5 Rxf7 Kg8 Rh3 Re8 Rg3 g5 Rf5 Nf4 Re3 Rh7 hxg5 hxg5 g3 Nd5 Rd3 Nf6 Rxg5 Rg7 Rf5 Ne4 a4 Rge7 f4 Re6 Kg2 R8e7 Kf3 Nd6 Rg5 Rg7 Rxg7 Kxg7 g4 a6 Rd5 Nc8 g5 Ne7 Rd7 Kf8 Rd8 Kg7 Ke4 Nc6 Rd7 Kf8 Kf5 Nxe5 fxe5 Re7 e6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c3 g6 Bg5 Bg7 Nd2 O-O e4 d6 f4 c5 dxc5 dxc5 Bc4 Nc6 Ngf3 Na5 Be2 Ng4 Nf1 Qb6 Qc1 c4 Bh4 e5 h3 Nf6 Bxf6 Qxf6 fxe5 Qe7 Qf4 Nc6 Ne3 Nxe5 Nxc4 Nxf3 Bxf3 Be6 Ne3 Qc5 Nc2 Rad8 Qe3 Qb5 Nb4 Bc4 a4 Qe5 Kf2 Bf6 Rhd1 Be7 Nc2 Rxd1 Rxd1 Bb3 Qd3 Qc5 Ke1 Bh4 Kd2 Rd8 Nd4 Bxd1 Bxd1 Bf6 b4 Qd6 Ke3 Qg3 Bf3 Qe1 Be2 Qg1 Kf3 Bxd4 cxd4 Qxd4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 Be7 Be2 O-O O-O b6 Be3 Ne5 Rc1 Bb7 f3 Rb8 Bd4 Re8 Qd2 Ba8 Rfd1 Nfd7 Bf1 Rc8 Kh1 Qc7 Qf2 Qb8 Be2 Ng6 Be3 Bh4 Qg1 Bf6 Nab1 Be5 Rd2 Bc6 Rdc2 Ba8 b4 h6 Nd2 Nf6 Bxb6 Nf4 Bf1 d5 exd5 exd5 Na4 Ne6 a3 Bf4 c5 Bc6 Nb2 Ng5 Bxa6 Nh5 Nf1 Bxc1 Rxc1 d4 Qf2 Nxf3 Bxc8 Re2 Re1 Rxf2
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 b6 g3 Bb7 d4 Bxf3 exf3 e6 Nc3 d5 cxd5 Nxd5 Qa4 c6 Bd3 Be7 Nxd5 exd5 Be3 O-O Rc1 a5 Kf1 Bd6 Qc2 h6 h4 Ra7 Kg2 Rc7 Bh7 Kh8 Bf5 Re8 Qd2 Bb4 Qd1 Rce7 a3 Bd6 Bg5 f6 Be3 Qc7 Bg6 Rf8 Qd3 Rc8 h5 Rf8 Rh4 a4 Bf5 b5 Qd2 Rfe8 Rg4 Kg8 Rc2 Nd7 Bxh6 Nb6 Be6 Kh7 Qd3 Kh8 Qg6 Rf8 Bxg7 Rxg7 Qh6 Rh7 Qxf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 Nbd2 c5 e4 cxd4 e5 Ng8 Bg2 Nc6 O-O Bb4 Qa4 Qc8 a3 Bxd2 Nxd2 f5 Bxc6 dxc6 Nb3 Kf7 Nxd4 c5 Nxe6 Kxe6 Rd1 Ne7 Rd6 Kf7 Bg5 Re8 Rd7 Kf8 Rad1 Bb7 e6 h6 Bxe7 Rxe7 Rxe7 Kxe7 Qd7 Qxd7 Rxd7 Kxe6 Rxb7 Kf6 f4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 Nxc3 bxc3 c5 O-O O-O Qc2 g6 e4 Nc6 Bh6 Re8 Rfd1 Qc7 Qe2 Red8 Qe3 e5 d5 Na5 c4 Nb3 Ra2 f6 h4 Bc8 Rb1 Nd4 Nxd4 cxd4 Qg3 Bf8 Bd2 Bd6 Rf1 Qg7 a4 a5 Rb2 Bc5 f4 Bd7 h5 Bxa4 h6 Qc7 f5 g5 Bxg5 fxg5 Qxg5 Kf8 Qf6 Ke8 Ra1 Qe7 Qe6 Rd6 Qg8 Qf8 Qg3 Qxh6 Rxa4 Qc1 Kf2 Qxb2 Kf3 Kf7
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 c4 c6 cxd5 cxd5 Ne5 Ng4 Nxg4 Bxg4 Nc3 Nc6 h3 Bd7 e3 e6 b3 Qa5 Bd2 Qc7 Rc1 Rac8 Qe2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 O-O c5 dxc5 Bxc3 bxc3 bxc5 Rb1 Qe7 Qb3 Bc6 Qa3 Rc8 Nd4 Rc7 Nb3 Nd7 Rfd1 Nb6 Bf3 Bb7 Na5 Rd8 Nxb7 Rxb7 h3 Rbd7 Rb5 Rc7 Qb3 Rcd7 a4 d4 exd4 cxd4 a5 Nc8 cxd4 Rxd4 Rxd4 Rxd4 Qc3 Rd8 Rb7 Qd6 Rc7 Ne7 Rxa7 Ng6 Qc7 Qf6 Qxd8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 f4 g6 Nxc6 bxc6 e5 Nd7 exd6 exd6 Qd4 Nf6 Be3 Be7 Be2 O-O Bf3 d5 Qd2 Qa5 O-O Bc5 Kh1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Qd2 Nc6 cxd5 exd5 Be2 Bf5 O-O Ne7 b4 c6 Rfc1 a6 a4 Qd6 Qb2 Rfe8 Qb3 Ng6 Ra2 Be7 b5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 Nbd2 Bb7 Bg2 Be7 O-O O-O Qc2 d5 cxd5 exd5 Ne5 c5 Ndf3 Nbd7 Bf4 Ne4 dxc5 Nxe5 Nxe5 Bxc5 Rad1 Qc8 Qb3 Qf5 Nd3 Rfd8 Nxc5 Nxc5 Qb4 Rac8 Be3 h6 Qa3 Qd7 Bd4 Qb5 Bh3 Rc7 Qf3 Ne6 Bxe6 fxe6 Qe3 Qe8 Rc1 Rdc8 Qe5 Rf7 g4 Rc6 f4 Qd7 h4 Qc7 Rxc6 Qxe5 Bxe5 Bxc6 Rc1 Bb7 h5 Rf8 Rc7 Rf7 Rc3 Kh7 Kf2 g6 hxg6 Kxg6 Ra3 Bc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 cxd4 exd4 O-O a3 Be7 d5 exd5 cxd5 Re8 Be3 d6 h3 Nbd7 Ng3 Bf8 Be2 Rxe3 fxe3 g6 O-O Qe7 Qd4 Bg7 Qf4 Ne8 Rac1 Be5 Qf2 Ndf6 Bd3 h5 Nge2 Nh7 Nf4 Nf8 Nb5 a6 Nd4 Bd7 Rc2 Bg7 Qg3 Rb8 Re2 Nf6 Nf3 Be8 e4 N6d7 Rc2 Ne5 Nxe5 Bxe5 Qf2 Nd7 b4 Qd8 Be2 Bg7 Nd3 Ne5 Nxe5 Bxe5 Rfc1 Kg7 Rc7 Qg5 R1c2 h4 Bg4 Kh6 Kh1 b6 Ra7 Bg3 Qd2 Bf4 Qd4 Be5 Qg1 a5 Qc1 Bf4 Qa1 Be5 Qg1 axb4 axb4 Bf4 Qa1 Be5 Qa3 Kg7 Rf2 Bf6 Qd3 Qe5 Be6 Kg8 Qf3 Kg7 Qf4 Qd4 Rf1 b5 Bg4 Rd8 Rc7 Qb2 Rc2 Qd4 Rd2 Qe5 Qxe5 Bxe5 Rc2 Bd7 Be2 f5 Rc7 Kh6 Bd3 fxe4 Bxe4 Kg5 Bd3 Bd4 Rb7 Bc3 Bxb5 Bf5 Be2 Ra8 b5 Ra2 Bf3 Rb2 Rb8 Bd4 Rd1 Bc5 Kh2 Be3 Re1 Bf2 Rf1 Bc5 Re8 Bd4 Rd1 Bf2 Be2 Bd7 Re4 Bf5 Re8 Bd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be2 Nf6 O-O Be7 Be3 O-O f4 e5 Nb3 exf4 Bxf4 Be6 Kh1 d5 e5 Nd7 Nxd5 Ndxe5 c4 Bg5 Nc5 Bxf4 Rxf4 b6 Ne4 b5 b3 bxc4 Bxc4 Kh8 Qh5 Bxd5 Bxd5 Qxd5 Rh4 h6 Rd1 Qa5 b4 Qxb4 Nf6 Qe7 Rf1 Rfd8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 Nc6 Qd2 a6 Nge2 Re8 Nc1 e5 d5 Nd4 N1e2 c5 dxc6 Nxc6 Nd5 b5 Bb6 Qd7 Nc7 Rb8 Nxe8 Qxe8 Be3 bxc4 Nc3 Be6 Be2 Nd4 O-O d5 exd5 Nxd5 Nxd5 Bxd5 Rf2 h5 Rc1 Qe6 Bf1 h4 Re1 Qc6 Bh6 Bh8 f4 e4 Rd1 Be6 f5 Nxf5 Qf4 Re8 Rfd2 Qc5 Kh1 Be5 Qg5 Kh7 Rd8 Rxd8 Rxd8 Qf2 Rd1 Nxh6 Qxe5 e3 Qc3 h3 Qe1 Ng4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O O-O c5 d4 d6 Nc3 Nc6 dxc5 dxc5 Be3 Be6 Qa4 Nd4 Rad1 Bd7 Qa3 Nc2 Qxc5 b6 Qg5 h6 Qf4 g5 Qe5 Rc8 Nd5 Nxd5 Qxd5 Be6 Qb7 Qc7 Qxc7 Rxc7 b3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 exd5 g3 Bd6 Bg2 O-O O-O Re8 Bg5 Nbd7 Nb5 h6 Nxd6 cxd6 Bxf6 Nxf6 Ne1 Qd7 Nd3 Qf5 Re1 Rac8 Qa4 Ra8 Rac1 Re7 Qb5 Rc8 Rxc8 Qxc8 Qb4 Qd8 Qa4 Qa8 Qc2 Qc8 Rc1 Qxc2 Rxc2 Kf8 h4 Ke8 Nf4 Kd8 Bf3 Rc7 Rxc7 Kxc7 g4 g5 hxg5 hxg5 Nh3 Nh7 e3 f6 Be2 Bc8 f4 Kd8 Kf2 Ke7 Kg3 Be6 Bd3 gxf4 exf4 Bg8 Bxh7 Bxh7 f5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 Nc3 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Bg4 Qa4 Na5 Rad1 Nc4 Bc1 Qc8 Qb5 Nb6 Bf4 Rd8 Rc1 Qd7 Qxd7 Rxd7 f3 Be6 b3 a6 Be3 Bd8 Nd1 Rc8 Nf2 Rxc1 Rxc1 Rc7 Rd1 Nbd7 Nd3 Rc3 Bd2 Rc8 Kf1 Bb6 Be3 Nf8 Kf2 Bd7 Rc1 Rxc1 Nxc1 Bc7 Nd3 Ng6 Bc1 Ne5 Nxe5 Bxe5 e3 b6 Ba3 a5 Be7 a4 Bd8 b5 Bf1 axb3 axb3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 cxd5 exd5 Ne2 Nbd7 O-O c6 f3 c5 a3 cxd4 exd4 Be7 Nf4 Nb8 g4 Bd6 Kh1 Re8 g5 Bxf4 Bxf4 Nh5 Bxb8 Rxb8 f4 g6 Qf3 b6 f5 Rb7 f6 Be6 Rae1 Qd6 Re5 Rd8 Qe3 b5 Be2 b4 axb4 Rxb4 Bxh5 gxh5 g6 hxg6 Rxe6 fxe6 Qh6 Rb7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 d4 cxd4 Nxd4 e6 g3 Qc7 Nd2 Bc5 N2b3 O-O Bg2 Be7 Qd3 a6 O-O d6 Bd2 Nc6 Nxc6 bxc6 Ba5 Qb7 Bc3 d5 Rac1 Bd7 Rc2 Rac8 Rfc1 Qa7 cxd5 cxd5 Bd4 Qa8 Bc5 Bxc5 Nxc5 Bb5 Qd4 Qb8 b3 Nd7 e3 Nxc5 Rxc5 Rxc5 Qxc5 Qe5 Qc3 Qf5 Qd4 h5 h4 Be2 e4 dxe4 Qxe4 Qxe4 Bxe4 Rd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O c4 d6 O-O Nbd7 Qc2 c5 Nc3 a6 h3 Rb8 Rd1 Qc7 Be3 b5 cxb5 axb5 Rac1 b4 Na4 c4 Qxc4 Qa5 b3 Ba6 Qc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 d6 O-O c5 dxc5 dxc5 Qxd8 Rxd8 Ne5 Ne8 Nd3 Nd6 Nxc5 Nc6 Na3 Rb8 Na4 Be6 Bf4 Rbc8 Rac1 Nd4 Rfe1 b5 Bxd6 Rxd6 Nxb5 Nxb5 cxb5 Rxc1 Rxc1 Rd2 Bf3 Bxb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6 Be2 exd5 e5 Ng4 cxd5 dxe5 h3 e4 hxg4 exf3 gxf3 Re8 f5 Qb6 Bh6 Qxb2 Bxg7 Kxg7 f6 Kg8 Qc1 Qb4 Kf1 Nd7 Bb5 Qd4 Kg2 Re3 Ne2 Qe5 Kf2 Rxe2 Bxe2 Nxf6 Qxc5 Bxg4 Qe3 Qxe3 Kxe3 Nxd5 Kf2 Be6 Rab1 b6 Rbc1 Nf4 a3 Nxe2 Kxe2 b5 Rc7 a5 Rb1 Bc4 Kf2 a4 Re1 Rd8 Re3 Rd2 Kg3 Kg7 f4 Rb2 Rc5 h5 Kh4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 c5 d5 exd5 cxd5 d6 Nf3 g6 Bf4 a6 a4 Bg7 e4 Bg4 Be2 O-O O-O Re8 Nd2 Bxe2 Qxe2 Nh5 Be3 Nd7 g4 Nhf6 f3 h6 Kh1 Ne5 Rg1 b5 g5 hxg5 Bxg5 c4 axb5 axb5 Rxa8 Qxa8 Nxb5 Qb8 Na3 Qxb2 Naxc4 Nxc4 Qxc4 Nh5 Qc6 Rb8 Qxd6 Be5 Qd7 Qd4 Qg4 Rb2 Rd1 Qd3 Qc8 Kg7 Qc4 Qa3 Qe2 Qa4 Rb1 Ra2 Be3 Qd7 Qf2 Qh3 f4 Bc3 Bd4 Bxd4 Qxd4 Kh7 Rg1 Kg8 d6 Rxd2 Qxd2 Qf3 Qg2 Ng3 hxg3 Qh5 Qh2 Qf3 Rg2 Qd1 Qg1 Qh5 Rh2 Qf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 b3 O-O Bb2 d6 e3 Nbd7 Qe2 a6 Rfd1 Re8 d4 Ne4 Nxe4 Bxe4 Nd2 Bxg2 Kxg2 Nf6 Rac1 Ra7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 c4 dxc4 Na3 c3 bxc3 c5 Nc4 Nc6 Nce5 Nd5 Bd2 Nxe5 Nxe5 Nb6 Bf4 Nd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 Nc6 Nge2 a6 Nc1 e5 d5 Nd4 Nb3 Nxb3 Qxb3 c5 dxc6 bxc6 O-O-O Qe7 Qb6 Bb7 g4 Rab8 h4 Rfc8 h5 Nd5 exd5 cxd5 Rxd5 Bxd5 Nxd5 Qe6 Qa7 Ra8 Qb7 Rab8 Qa7 Ra8 Qb7 Rab8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nf6 g3 b6 Bg2 Bb7 O-O e6 Nc3 Be7 b3 O-O Bb2 d6 e3 Nbd7 d4 a6 Qe2 Ne4 Rfd1 Qb8 Nxe4 Bxe4 Ne5 Bxg2 Nxd7 Qb7 Nxf8 Bf3 Qd3 Rxf8 Rd2 f5 Re1 Qc8 Qc3 Rf6 a3 Qe8 dxc5 Qh5 h4 Qg4 Kh2 bxc5 Rh1 Rg6 Kg1 Bxh4 Qa5 h6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 g6 d4 Bg7 g3 Nf6 Bg2 O-O c4 c5 O-O cxd4 Nxd4 Nc6 Nc3 Nxd4 Qxd4 d6 Qd3 a6 Be3 Bd7 Bd4 Bc6 e4 Re8 Rfe1 Rc8 Rad1 Qa5 a3 b5 cxb5 axb5 e5 dxe5 Bxc6 exd4 Bxe8 dxc3 Bd7 Rd8 Qxb5 Qc7 bxc3 Rxd7 Rxd7 Qxd7 Qxd7 Nxd7 Rxe7 Nb6 Rb7 Na4 Rb8 Bf8 c4 Kg7 Kg2 Bd6 Ra8 Nb2 a4 Nxc4 a5 Ne5 Rc8
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6 Bg5 a6 Bxf6 Bxf6 Qf4 O-O Rfd1 Be7 Ne4 Bxe4 Qxe4 Ra7 Nd4 Qc8 b3 Re8 a4 Qc5 Ra2 Bf6 Rad2 Rc7 Qb1 Be7 b4 Qh5 Rc2 Rec8 b5 axb5 axb5 Rxc4 Rxc4 Rxc4 Qa2 Qc5 Qa8 Rxd4 Qxb8 Bf8 Ra1 d5 Bf1 Rc4 Ra8 Rc1 Qe8 d4 Ra7 Qf5 Ra8 Qc5 g4 Qd6 Rd8 Qb4 Rd7 h6 Qxf7 Kh7 g5 Qb1 g6 Qxg6 Qxg6 Kxg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 g6 d4 exd4 c3 Bg7 cxd4 b5 Bc2 d6 d5 Ne5 Nxe5 dxe5 a4 Bd7 Be3 Nf6 O-O O-O Qd2 Ne8 Bc5 Nd6 axb5 Bxb5 Re1 Qd7 Nc3 Rfb8 b4 Bf8 Qg5 Qe7 Qe3 Qf6 Bd3 Bxd3 Qxd3 Nb5 Nxb5 axb5 Rxa8 Rxa8 Qxb5 Qa6 Qxa6 Rxa6 g4 Bd6 b5 Ra8 Bxd6 cxd6 b6 Rb8 Rb1 Kf8 Kf1 Ke7 Ke2 g5 Kd3 Kd7 Kc4 Rc8 Kb5 Rc2 Ra1 Rb2 Ka6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 Be7 Bd3 d5 O-O Nc6 Re1 Bf5 Nbd2 Nxd2 Qxd2 Bxd3 Qxd3 O-O c3 Qd7 Bf4 a6 Re3 Rae8 Rae1 Bd8 h3 Rxe3 Rxe3 f6 Re2 Rf7 Nd2 Be7 Nf1 Bf8 Qf3 Re7 Ne3 Nd8 Bxc7 Qxc7 Nxd5 Qd6 Nxe7 Bxe7 Qe4 Bf8 Qe8 g6 a4 Kg7 b4 Qc7 Re3 Nf7 Qe6 Qd8 a5 h5 Qe4 Qd7 Qe6 Qd8 Kf1 Nh6 g4 hxg4 hxg4 Nf7 Ke2 Ng5 Qb6 Qd7 Kd3 Bd6 Kc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Qa5 Qd2 Nc6 Rc1 cxd4 cxd4 Qxd2 Kxd2 O-O d5 Rd8 Ke1 Na5 Bg5 Bf6 Bd2 b6 Rc7 Bg4 Ba6 e6 Ng5 Be5 Rxf7 exd5 f4 Bg7 f5 dxe4 Bxa5 bxa5 Bc4 Bc3 Kf2 e3 Kg3 Be5 Kxg4 Rd4 Kh3 Rxc4 f6 Bxf6 Rxf6 Re8 Re1 e2 Kg3 Ra4 Kf2 Rxa2 Ne6 a4 Rb1 a3 Rb7 e1=Q Kxe1 Rxg2 Rg7 Kh8 Rgf7 h5 Kf1 Rxh2 Rxg6 Rxe6 Rxe6 Kg8 Rxa7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 dxc4 Nf3 Nf6 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Nh4 Bg6 g3 O-O Qb3 Qb6 Nxg6 hxg6 Rd1 a5 Qc2 c5 Nb5 cxd4 exd4 Qc6 Qe2 Nb6 Bb3 Qe4 Be3 Nbd5 Bc2 Qg4 f3 Qh3 Bf2 g5 Bd3 g6 Qe5 Kg7 Bf1 Qf5 Bg2 Rfc8 Rac1 Kg8 h3 Be7 Qe2 Nd7 Nc3 N7b6 Ne4 Nxa4 Be1 Rc6 b3 Nab6 Rxc6 bxc6 Rc1 Nb4 f4 gxf4 g4 Qb5 Qxb5 cxb5 Bxb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 Ne4 Nxe4 Bxe4 Nd2 Bg6 g3 Nc6 e3 a6 b4 b5 cxb5 axb5 Bb2 Na7 h4 h6 d5 exd5 Bg2 c6 O-O f6 Re1 Be7 Qg4 Kf7 h5 Bh7 e4 dxe4 Bxe4 Bxe4 Nxe4 Nc8 Rad1 Ra7 Nxf6 gxf6 Qg6 Kf8 Bc1 d5 Rd4 Nd6 Rg4 Nf7 Bxh6 Ke8 Bg7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N5c3 Be7 Be2 O-O O-O b6 Bf4 Bb7 Re1 Ne5 Nd2 Nfd7 Bf1 Bg5 Bxg5 Qxg5 Nf3 Qf6 Nxe5 Nxe5 Qd2 g5 Rad1 Rfd8 Qe3 Rd7 Rd2 Bc6 Red1 Rad8 f3 h5 Be2 h4 Rf1 Ng6 g3 hxg3 hxg3 Qe5 Kf2 Re8 Qd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: g3 g6 Bg2 Bg7 d4 d5 Nf3 Nf6 c4 dxc4 Na3 c3 bxc3 O-O O-O c5 Ne5 Nc6 Nac4 Nd5 Bb2 Be6 Qc2 Rc8 Nxc6 Rxc6 Ne5 Rc8 Qa4 a6 Rfd1 Qc7 Nd3 Nb6 Qa5 Nc4 Qxc7 Rxc7 Rab1 cxd4 cxd4 Nxb2 Rxb2 b5 Nb4 Bxd4 Nxa6 Bxb2 Nxc7 Bxa2 Rd2 Be5 Nxb5 Bc4 Na7 Rb8 Bf3 Rb1 Kg2 g5 Nc6 Bf6 Nd4 e6 h3 Kg7 e3 Bf1 Kh2 Ba6 Kg2 Be5 Be2 Bb7 Bf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 d6 Nc3 e5 Nf3 Nbd7 Bg5 Be7 e3 O-O Qc2 c6 Bd3 h6 h4 b5 dxe5 Nxe5 Nxe5 dxe5 O-O-O Qa5 Kb1 Be6 Bxf6 Bxf6 Ne4 Be7 Ng5 bxc4 Bh7 Kh8 Bf5 Bxf5 Qxf5 Bxg5 hxg5 Rab8 Rc1 Qd2 Rc2 Qd3 g4 Qxf5 gxf5 Rb4 Rh4 Rd8 a3 Ra4 gxh6 g6 fxg6 fxg6 Rhxc4 Rxc4 Rxc4 Rd6 Rc5 Re6 Ra5 Re7 Ra6 Rc7 Kc2 Kh7 Kd3 Rf7 Ke4 Rxf2 Rxa7 Kxh6 b4 g5 Ra6 Kh5 Rxc6 Ra2 Rc1 g4 Kxe5 Rxa3 Kf4 Rb3 Rc5 Kh4 Rc8 Rxb4 e4 Kh3 Rh8 Kg2 Rg8 Kh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Be2 O-O O-O a6 a4 Bg4 h3 Bxf3 Bxf3 Nbd7 Bf4 Qc7 a5 Rfe8 Qc2 c4 Ra4 Ne5 Bxe5 Rxe5 Nb1 Rae8 Na3 Nxe4 Rxc4 Qe7 Rc7 Qd8 Rxb7 Ng5 Bg4 h5 Bd7 Nf3 gxf3 Qg5 Bg4 hxg4 fxg4 Re1 Qc4 Qd2 Nc2 R1e4 Qb3 Be5 Kg2 Re2 Ne3 Bd4 Nd1 R2e4 Rb8 Rxb8 Qxb8 Kg7 Qxd6 Be5 Qa3 Rf4 Qe3 Qxd5 f3 Rb4 Rf2 Qxa5 Re2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nf3 d6 O-O c5 Nc3 Nc6 d5 Na5 Nd2 a6 Qc2 Rb8 b3 b5 Bb2 bxc4 bxc4 Bh6 Ncb1 e5 Bc3 Bd7 Na3 Rb4 Bxb4 cxb4 Nab1 Qc7 e3 Bf5 Ne4 Bxe4 Bxe4 Nb7 Nd2 Nc5 Bg2 Rb8 Rfb1 a5 a3 e4 axb4 axb4 Bh3 Bg7 Ra2 h5 Nb3 Nd3 Rd1 Ne5 c5 Nd3 cxd6 Qxd6 Bf1 Ne5 Ra6 Qd7 Rxf6 Bxf6 Qxe4 Rc8 Qxb4 h4 Qf4 Kg7 gxh4 Qd6 Nd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 Nf3 Nf6 e3 Bg4 Bxc4 e6 h3 Bh5 Nc3 a6 g4 Bg6 Ne5 Nbd7 Nxg6 hxg6 Bf1 c6 Bg2 Qc7 O-O Be7 f4 Nb6 g5 Nfd7 Qg4 O-O-O Rb1 Kb8 b4 Nd5 Na4 f5 Qg3 Nxb4 Bd2 Nd5 Rfc1 Ka7 Qe1 Ba3 Rc2 Qd6 Rb3 Qe7 Qe2 Rb8 Qd3 Bd6 Nb2 Rhc8 Nc4 Bc7 a4 b5 axb5 cxb5 Ra2 Kb7 Bb4 Qe8 Bd6 Ra8 Qb1 Kc6 Rba3 bxc4 Rxa6 Rxa6 Rxa6 Bb6 Bc5 Qd8 Qa1 Nxc5 dxc5 Kxc5 Ra4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Bb4 cxd5 exd5 Bg5 h6 Bh4 c5 e3 g5 Bg3 Ne4 Bb5 Kf8 dxc5 Nxc3 bxc3 Bxc3 Ke2 Bxa1 Qxa1 f6 h4 g4 Nd4 Kf7 Bd3 Nd7 Qc3 Ne5 Nb5 Qe7 Nd6 Kf8 Rd1 b6 Bc2 Ba6 Ke1 Bc4 Nxc4 dxc4 Rd6 Re8 Bxe5 fxe5 Qxc4 Qf7 Qe4 g3 fxg3
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6 b3 O-O e4 Nbd7 Qe3 a6 Nd4 Qc7 Bb2 Rfe8 h3 Bf8 Rfe1 Rad8 Re2 g6 Rae1 Qb8 Qd2 Bg7 Qd1 Nc5 Bc1 Rc8 Bg5 h6 Bc1 Red8 Bb2 b5 cxb5 d5 exd5 Nxd5 Nxd5 Bxd5 b4 Bxg2 Kxg2 e5 bxc5 exd4 Rd2 Rxc5 bxa6 Qa8 Qf3 Qxa6 Red1 Rf5 Qe4 Qa4 a3 Re8 Qb7 Rd8 Rd3 h5 R1d2 Qe8 Kf1 Rb8 Qc7 Rxb2 Rxb2 Qe4 Qc4 Qh1 Ke2 Qg1 Rb8 Kh7 f4 h4 Rb5 Rxb5 Qxb5 hxg3 Qg5 Qf2 Kd1 Qf1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 g6 Nc3 d6 e4 Bg7 Be2 O-O Bg5 b5 cxb5 a6 a4 Qa5 Bd2 Nbd7 Ra3 Bb7 Nf3 axb5 Bxb5 Qc7 O-O Ng4 Bg5 Ngf6 Nd2 e6 Nc4 exd5 exd5 Rad8 a5 h6 Bh4 Ne5 a6 Ba8 Re1 g5 Bg3 Nfd7 a7 f5 Bxe5 Nxe5 Nxe5 dxe5 Ra6 e4 Bc4 Qf7 Nb5 Kh7 Re6 Qb7 Qh5 Rf6 Rxf6 Bxf6 g4 f4 h4 Kg7 Nc3 e3 Bd3 exf2 Kxf2 Qxb2 Re2 Qxc3 Qg6 Kf8 Qxh6 Bg7 Qxg5 Qf6 Qxf6 Bxf6 Bc4 Bxh4 Kf3 Rd7 Ra2 Bg5 Ke4 Rf7 Ra5 Kg7 Rxc5 Kf6 Rc8 Rxa7 Rf8 Kg7 Rc8 Kf6 Rf8 Kg7 Rc8 Bb7 Rc7 Kf6 Kd4 Bh4 d6 Bf2 Kc3 Be4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 Nc6 Nge2 a6 Qd2 Rb8 h4 b5 h5 e5 d5 Na5 Ng3 bxc4 O-O-O Nd7 hxg6 fxg6 Nb1 Rb5 b4 cxb3 Bxb5 c5 dxc6 axb5 Qd5 Rf7 axb3 Nf8 Qxd6 Qe8 Qd8 Qxc6 Kb2 Qa8 Rc1 Nc4 bxc4 Rd7 Qe8 bxc4 Nc3 Qc6 Kc2 Rd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 d6 Nc3 e5 Nf3 Nbd7 e4 Be7 Be2 O-O O-O c6 d5 Nc5 Qc2 cxd5 cxd5 Qc7 Nd2 Bd7 a4 Rac8 Ra3 Ne8 Qd1 Bg5 b4 Na6 Qb3 f5 Nc4 Bxc1 Rxc1 fxe4 Nxe4 Bf5 Ng3 Qd7 h3 Nf6 Raa1 Bg6 Ne3 Rxc1 Rxc1 Rc8 Rc6 bxc6 dxc6 Qf7 Bc4 d5 Nxd5 Kh8 Nb6 Qc7 Nxc8 Qxc8 b5 Nc5 Qa3 Nce4 Qe7 Nxg3 c7 Bf5 fxg3 h5 a5 e4 b6 axb6 axb6 Kh7 Qc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 d5 Nc3 Be7 cxd5 exd5 g3 O-O Bg2 c5 O-O Bb7 Qc2 Nbd7 Be3 Re8 Rfd1 Bf8 Bg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 h6 Bf4 e6 dxe6 Bxe6 Bxd6 Re8 Nf3 Nc6 O-O Nd4 e5 Nd7 Nxd4 cxd4 Qxd4 Nxe5 Bxe5 Qxd4 Bxd4 Bxd4 Rac1 Rad8 b3 Bxc3 Rxc3 Rd2 Bf3 Rxa2 Bxb7 Rb8 Bf3 Ra3 Bd1 a5 Rc1 Rc8 Re1 a4 bxa4 Rxc4 Rxc4 Bxc4 Bc2 Bd5 h4 Bc6 Rd1 Ra2 Bb3 Ra3 Rd6 Rxb3 Rxc6 Rb1 Kh2 Rb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 g3 Bg7 Bg2 Nf6 Nf3 O-O O-O c5 Nc3 Nc6 d4 d6 dxc5 dxc5 Bf4 Be6 Ne5 Na5 Nd3 Nh5 Nxc5 Nxf4 gxf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 c5 d5 Ba6 Qc2 exd5 cxd5 g6 Bf4 d6 Nc3 Bg7 Qa4 Qd7 Bxd6 Qxa4 Nxa4 Nxd5 O-O-O Ne7 e4 Bxf1 Rhxf1 Nbc6 Nc3 Rd8 Nb5 Rd7 Bf4 Rxd1 Rxd1 O-O Rd7 Ra8 Bd6 Nc8 Nc7 Rb8 Na6 Ra8 Bf4 N8e7 Bd6 Nc8 Bg3 N8e7 Bh4 Bf8 Bf6 Rd8 Rc7 Re8 g4 Bg7 g5 Bf8 Kc2 Rc8 Rb7 Nd4 Nxd4 cxd4 Kd3 Nc6 f4 Bd6 e5 Bf8 b4 Nd8 Bxd8 Rxd8 Rxa7 h6 h4 h5 Nc7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 e6 g3 d5 exd5 exd5 Bg2 Nf6 Nge2 d4 Ne4 Nxe4 Bxe4 Nd7 O-O Nf6 Bg2 Bd6 c3 d3 Nf4 O-O Nxd3 Bxg3 fxg3 Qxd3 Qf3 Qxf3 Bxf3 Bh3 Bxb7 Rae8 Bg2 Bxg2 Kxg2 Re2 Rf2 Rfe8 b3 Rxf2 Kxf2 Ng4 Kg2 f5 h3 Ne5 d4 cxd4 cxd4 Nd3 Bg5 h6 Rd1 hxg5 Rxd3 Re2 Kf3 Rxa2 d5 Kf7 d6 Ke8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Qe2 Be7 a4 Nc6 Be3 O-O O-O d5 Rfd1 Bd6 exd5 exd5 Nxd5 Nxd5 Bxd5 Bxh2 Kxh2 Qxd5 c4 Qe4 Nxc6 bxc6 Rd4 Qe5 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 Nxc3 bxc3 c5 O-O Nc6 e4 O-O Be3 cxd4 cxd4 Rc8 Qe2 Na5 Rfe1 Qd6 d5 exd5 e5 Qe6 Nd4 Qxe5 Nf5 Bf6 Qg4 Rce8 Bd2 Qxa1 Rxa1 Bxa1 Nxg7 Bxg7 Bh6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 d4 Nf6 e4 d6 Nf3 O-O h3 e5 d5 Na6 Be3 Nh5 Nh2 Qe8 Be2 Nf4 Bf3 f5 h4 Qe7 g3 Nb4 Qb3 Nfd3 Ke2 f4 Bd2 fxg3 fxg3 Rxf3 Nxf3 Bg4 Raf1 Rf8 Nd1 Qf7 Be3 Bxf3 Kd2 Qd7 Rhg1 Qh3 a3 Bxe4 Rxf8 Bxf8 axb4 Qh2 Kc3 Nc1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Qe7 g3 Bxd2 Qxd2 O-O Bg2 d5 O-O dxc4 Na3 c5 dxc5 Qxc5 Rac1 Nc6 Nxc4 Qe7 Nfe5 Nxe5 Nxe5 Nd5 Rfd1 Nb6 Qa5 g6 Rd3 Nd5 e4 Nb6 Bf1 Re8 Rdd1 Rf8 a3 Kg7 b3 Kg8 a4 Rd8 Qc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Bxd2 Qxd2 d5 Bg2 c6 Qb2 O-O Nbd2 Nbd7 O-O c5 Rac1 Bb7 b4 cxb4 Qxb4 Rc8 Rfe1 Nb8 Qa4 dxc4 Nxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 Nxc3 bxc3 c5 O-O Nc6 e4 Rc8 Bb2 Bf6 d5 exd5 exd5 Qxd5 Re1 Kf8 Qc2 Ne5 Be4 Nxf3 Bxf3 Qd7 Rad1 Qc7 Bxb7 Qxb7 Qa4 h5 Rd7 Qc6 Qxa7 Kg8 Rb7 Kh7 Rxb6 Rhe8 Rf1 Qc7 Rb7 Qc6 Rxf7 Ra8 Qc7 Qxc7 Rxc7 Re2 Bc1 Ra5 Be3 Rxa3 Rxc5 Kg6 Rc1 Raa2 g3 Rac2 Rxc2 Rxc2 Bd4 h4 Bxf6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Be7 O-O b5 a3 O-O Be3 Bb7 f3 Nc6 Nxc6 Bxc6 Qd2 Nd7 Rfd1 Qc7 Qf2 Rfe8 Ne2 Ne5 Nd4 Bd7 c3 Nc4 Bc1 Bf8 h3 Qb7 Bc2 Rac8 Kh1 g6 Nb3 Bg7 Nd2 d5 Nxc4 bxc4 Rb1 Bc6 exd5 Bxd5 Be3 Red8 Bb6 Rd7 Ba4 Bc6 Rxd7 Bxd7 Bc2 Bc6 Be3 Bd5 Kg1 Qb8 Qe2 Rd8 Bf2 h5 Qe1 Rd7 Ba4 Rb7 Qd2 Be5 Bd1 Bh2 Kf1 Bd6 Bd4 Bxa3 Ra1 Bd6 Rxa6 Bf4 Be3 Bxe3 Qxe3 Rxb2
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 d4 Bg7 Nc3 Nf6 e4 d6 Be2 O-O Bg5 Nbd7 Qc1 c5 d5 b5 cxb5 a6 a4 Qa5 Bd2 axb5 Nxb5 Qb6 Qc2 Ba6 Nf3 Bxb5 Bxb5 Qxb5 axb5 Rxa1 Bc1 Nxe4 O-O Nef6 b4 Nxd5 Bd2 Rfa8 bxc5 Rxf1 Kxf1 Ra1 Ke2 Nxc5 Qc4 e6 b6 Nxb6 Qb5 Nbd7 Be3 Bf8 Nd4 Ra2 Kf1 Ra1 Ke2 e5 Nc6 Ra2 Kf1 Ra1 Ke2 Ra2 Kf1 Ra6 Bxc5 Nxc5 g3 Ra1 Kg2 Ne6 Qb8 Rd1 Qb2 Rd5 Qb8 Rc5 Ne7 Kg7 Nc8 Rd5 Qa8 Rd2 Nb6 Nc5 Nc4 Rd4 Ne3 Be7 h4 h5 Nd5 Bd8 Kf3 Ne6 Qc6 Rd2 Ke3 Re2 Kd3 e4 Kc4 Rc2 Nc3 Bf6 Qxe4 Rxc3 Kd5 Rc5 Kxd6 Be5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Qc2 Bb7 a3 Bxc3 Qxc3 d6 e3 Nbd7 b4 O-O Bb2 Qe7 Bd3 c5 dxc5 bxc5 Be2 d5 cxd5 Bxd5 O-O cxb4 axb4 Rfc8 Qd4 Nb6 Rfc1 Rxc1 Rxc1 Rc8 Rxc8 Nxc8 Ne5 Nd6 f3 Qc7 e4 Ba8 b5 Nfe8 Qa4 f5 Qb3 Qc8 exf5 Qc5 Kf1 Bd5 Qa3 Qxa3 Bxa3 exf5 Bc5 Nc8 Nc6 Kf7 Nxa7 Nxa7 Bxa7 Ke6 Bd4 g6 Kf2 Nd6 Ke3 g5 g3 Nc4 Kd3 Nd6 Kc3 f4 Kb4 fxg3 hxg3 h5 Bf2 Nf5 f4 gxf4 gxf4 Ng7 Kc5 Bg2 Bd4 h4 Bxg7 h3 Bg4 Ke7 Bxh3 Bxh3 Kb6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6 Bg5 a6 Bxf6 Bxf6 Qd3 Ra7 Rad1 Be7 Ne4 Bxe4 Qxe4 O-O Nd4 Qc8 b3 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 Qc2 c5 e4 Nxc3 bxc3 Nd7 Bd3 Qc7 Qd2 g6 O-O Bg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5 Re8 Nd5 Nxd5 cxd5 c6 Bc4 cxd5 Bxd5 Nc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 c5 Nc3 cxd4 Nxd4 d5 cxd5 Nxd5 Bg2 Nxc3 bxc3 O-O O-O Qa5 Be3 Nc6 Qb3 Qa6 Rab1 Rd8 Qb2 Nxd4 Bxd4 e5 Bc5 Rb8 Rfd1 Bf5 e4 Be6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 c5 d5 exd5 cxd5 d6 e4 g6 f4 Bg7 Bb5 Nfd7 a4 Na6 Nf3 Nb4 O-O a6 Bxd7 Bxd7 f5 O-O Bg5 f6 Bf4 gxf5 Bxd6 Bxa4 Rxa4 Qxd6 Nh4 fxe4 Nf5 Qd7 Nxe4 Kh8 Nxc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 c5 d5 d6 Nc3 O-O Nf3 e6 O-O exd5 cxd5 a6 a4 Re8 Nd2 Nbd7 h3 Rb8 Nc4 Ne5 Na3 Nh5 e4 Rf8 Kh2 f5 f4 b5 axb5 axb5 Naxb5 fxe4 Bxe4 Bd7 Qe2 Qb6 Na3 Rbe8 Bd2 Qxb2 fxe5 Bxe5 Nc4 Nxg3 Rxf8 Rxf8 Qe1 Nxe4 Kg2 Qc2 Nxe5 Rf2 Qxf2 Nxf2 Ra2 Qf5 Nxd7 Nd3 Bh6 Qxd7 Ra8 Kf7 Rh8 Kf6 Kf3 Qxh3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 Bf4 a6 a4 Bg7 e4 Bg4 Be2 O-O O-O Qe7 Nd2 Bxe2 Qxe2 Nh5 Be3 Nd7 a5 Bd4 Ra4 Qf6 Qd3 Ne5 Bxd4 Nxd3 Bxf6 Nxf6 Nc4 Rad8 Rd1 Nb4 Rd2 Ne8 Ra1 Nc7 Re1 Kg7 b3 Rfe8 g4 Rd7 f3 Nb5 Ne2 f5 gxf5 gxf5 Ng3 fxe4 fxe4 Kh8 Rf1 Nd4 Rg2 Nxb3 Nf5 Rf8 Nfxd6 Rxf1 Kxf1 Nxa5 Ne5 Rg7 Nef7 Kg8 Nh6 Kf8 Rf2 Ke7 Nhf5 Kd7 Nxb7 Nd3 Nxa5 Nxf2 Kxf2 Rg4 Kf3 Rg1 e5 Rf1 Ke4 Re1 Kf4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Bxd2 Qxd2 O-O Nc3 d5 e3 Qe7 Rd1 Rd8 Qc2 Nbd7 Be2 dxc4 Bxc4 c5 O-O Nb6 Be2 Bd7 e4 cxd4 Rxd4 e5 Rd2 Bc6 Rfd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 O-O Be3 b6 h4 Bb7 Qf3 Qd7 Ne2 h5 Bg5 Nc6 Nf4 e6 Rd1 Na5 Bd3 e5 dxe5 Bxe5 O-O Qg4 Qe3 Rfe8 Be2 Bxf4 Bxf4 Nc4 Bxc4 Rxe4 f3 Qxf4 Bxf7 Kg7 Qd3 Qe3 Qxe3 Rxe3 Rd7 Kh6 Rxc7 Ba6 Rd1 Bd3 Rd2 Bf5 Kf2 Re5 Rd5 Rxd5 Bxd5 Rd8 c4 b5 Ke3 a5 Kf4 Bb1 g4 hxg4 fxg4 Rf8 Kg3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 c5 Nc3 cxd4 Nxd4 d5 cxd5 Nxd5 Bd2 Be7 e4 Nb4 Be3 O-O Be2 N8c6 Nxc6 Nxc6 O-O Qa5 Qb3 Bc5 Bxc5 Qxc5 Rfd1 e5 Qc4 Qe7 Nd5 Qg5 Qc1 h6 Qxg5 hxg5 Rac1 Nd4 Kf1 Rb8 Ne3 Bd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 Qc2 c5 e4 Nxc3 bxc3 Nc6 Bb2 Rc8 Rd1 cxd4 cxd4 a6 Qd2 Na5 d5 exd5 exd5 Bd6 Bxg7 Qe7 Be2 Rg8 Qh6 f5 Bf6 Qf8 Qxh7 Qf7 Qxf5 Rg6 Qe4 Kf8 Ng5 Rxg5 Bxg5 Re8 Bh6 Kg8 Qg4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Be7 O-O Nc6 Be3 O-O f4 e5 fxe5 dxe5 Nf5 Bxf5 Rxf5 Qa5 Kh1 Rad8 Qf1 Qb4 Rb1 Qd6 Bc4 Nd4 Bxd4 Qxd4 Rd1 Qc5 Nd5 Kh8 Qe2 Nxd5 Bxd5 g6 Rf3 f5 c4 fxe4 Rxf8 Rxf8 Qxe4 Qf2 Qxe5 Bf6 Qe1 Qxb2 Rd2 Qb4 Rd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 c6 Bg5 h6 Bh4 dxc4 e4 g5 Bg3 b5 Be2 Bb7 e5 Nd5 h4 Qa5 Rc1 g4 Nd2 c5 Nce4 cxd4 O-O h5 a4 a6 b4 Qd8 Bxc4 bxc4 Nxc4 Nc3 Nxc3 dxc3 Nd6 Bxd6 exd6 Qf6 Qd3 O-O Rxc3 Bd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 a4 h6 Nbd2 Bf8 Bc2 exd4 cxd4 Nb4 Bb1 bxa4 Rxa4 a5 Ra3 g6 e5 dxe5 dxe5 Nh5 Nh2 Qd5 Ndf3 Rxe5 Ng4 Rxe1 Qxe1 Kh7 Re3 Rd8 Bd2 Qd6 Nge5 Bd5 Nh4 Kg8 Nexg6 fxg6 Bxg6 Bf7 Bxh5 Bxh5 Rg3 Kf7 Qe4 Qxd2 Qf5 Ke7 Re3 Qxe3 fxe3 Rd1 Kh2 Rd5 Qc8 Kf7 g4 Bd6 Kg2 Bg6 Nxg6 Kxg6 Qg8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6 dxe6 fxe6 Bd3 Nc6 O-O Nd4 Ng5 e5 f5 h6 Nh3 gxf5 exf5 b5 Be3 bxc4 Bxc4 Kh8 Bxd4 cxd4 Nd5 Ba6 Nxf6 Bxc4 Nh5 Bxf1 Qg4 Qd7 Rxf1 d3 Qf3 d2 g4 Rac8 Qd3 Qa4 Nf2 Qd4 Qxd4 exd4 Nf4 Rfe8 Ne6 Rc1 Nd1 Bf6 Kf2 Bg5 Ke2 Rc5 Kd3 Re5 Nxg5 hxg5 Rf2 Re4 h3 Re3 Kxd4 R8e4 Kd5 Re2 Rf3 Re1 f6 Rf4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Qe7 g3 Nc6 Nc3 d5 cxd5 exd5 Bg2 O-O O-O Re8 Qb3 Bg4 e3 a5 a3 Bxf3 Bxf3 a4 Qd1 Bxc3 Bxc3 Qd7 Rc1 Na5 Bxa5 Rxa5 Rc5 Rea8 Qc2 c6 Rc1 g6 Qd3 Kg7 R1c3 h5 h4 Qe7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 Nbd7 O-O e5 Re1 h6 Qc2 Nh7 dxe5 dxe5 Be3 Re8 Rad1 Nhf8 c5 Ne6 c6 bxc6 Na4 g5 Qxc6 Rb8 h3 h5 Qc1 g4 hxg4 hxg4 Nh2 g3 fxg3 Nd4 Bc4 Nb6 Nxb6 Rxb6 Nf3 Be6 Bxd4 exd4 e5 c5 Bxe6 Rbxe6 Qxc5 d3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 Nc3 c6 Bg5 Nbd7 e3 Qa5 Nd2 dxc4 Bxf6 Nxf6 Nxc4 Qc7 Be2 Be7 O-O O-O Rc1 Rd8 Qc2 Bd7 Ne4 Nxe4 Qxe4 c5 dxc5 Bc6 Qe5 Qxe5 Nxe5 Rd2 Bf3 Bxf3 gxf3 Rxb2 Rb1 Rxb1 Rxb1 Bxc5 Rxb7 f6 Nd7 Bd6 f4 a5 Rb6 Bb4 Nb8 Kf7 a4 Bc3 Rb7 Kg6 Kg2 Bb4 Nc6 Bc3 Ne7 Kh6 f5 e5 Kf3 Rd8 Ke4 Be1 f3 g6 Rb6 Kg5 fxg6 hxg6 f4 exf4 exf4 Kh5 Rxf6 Rd2 Nxg6 Rxh2 Kf5 Rc2 Rf7 Rc5 Ne5 Kh6 Rf6 Kh7 Rc6 Rxc6 Nxc6 Bd2 Ne5 Kg7 Nc4 Bc3 Ke6 Kf8 f5 Ke8 f6 Bb4 f7 Kf8 Kf6 Bc3 Kg6 Bb4 Ne5 Ke7 Kg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 Ne4 Bf4 c5 d5 Qb6 Bc1 e6 f3 Qa5 c3 Nf6 e4 d6 Na3 exd5 exd5 Be7 Nc4 Qd8 Ne3 O-O Ne2 Re8 g4 Nfd7 Ng3 Bg5 Kf2 Ne5 Bb5 Bd7 Bxd7 Nbxd7 Nef5 c4 Nh5 Nd3 Kg3 Bxc1 Rxc1 g6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 Qc2 c5 e4 Nxc3 bxc3 Be7 Bb5 Bc6 Bd3 Nd7 O-O h6 Rd1 Qc7 d5 exd5 exd5 Bxd5 Bb5 a6 Bf4 Qxf4 Bxd7 Kxd7 Rxd5 Kc7 Re1 Bd6 Rf5 Qc4 Re4 Qb5 Rxf7 Kb8 Re6 Rd8 c4 Qc6 Ne5 Qc8 Qb1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 e5 Bf5 Nc3 Qb6 g4 Bd7 Na4 Qc7 Nc5 e6 Nxd7 Nxd7 f4 c5 c3 Ne7 Nf3 h5 f5 hxg4 fxe6 gxf3 exd7 Qxd7 Qxf3 cxd4 cxd4 Nc6 Be3 Bb4 Kf2 Rh4 Rd1 Rc8 Rg1 Rxh2 Rg2 Rxg2 Kxg2 Nd8 Bd3 Ne6 Rf1 a6 Qh5 g6 Qh8 Nf8 Kg3 Be7 Qg8 Qe6 Qh8 f5 Qh3 Qf7 Qh1 Ne6 Qxd5 Rd8 Qh1 Nxd4 Qh8 Bf8 Bg5 Rc8 e6 Nxe6 Re1 Kd7 Qh4 Bd6 Kf2 Nxg5 Qxg5 Qh7
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 e3 g6 Bb5 c6 Bd3 Bg7 e4 Nxc3 bxc3 c5 Bg5 Qd6 e5 Qd7 dxc5 O-O cxb6 axb6 O-O Qc7 Bb5 Bxe5 Bh6 Bg7 Bxg7 Kxg7 Qd4 Kg8 Ng5 h6 Ne4 Bxe4 Qxe4 Na6 Qe3 Qc5 Qxc5 Nxc5 Rfb1 Rfd8 Rb4 Rd6 Bf1 Kf8 a4 Ra5 g3 Ke7 Kg2 f5 Bb5 Rd2 Rd4 Rxd4 cxd4 Nxa4 Rxa4 Rxb5 Ra7 Kd6 Rh7 h5 Rg7 Rd5 Rxg6 b5 Kf3 b4 Ke3 b3 Kd2 Rxd4 Kc3 b2 Kxb2 Rd2 Kc3 Rxf2 h4 f4 Rg5 Rf3 Kd4 Rxg3 Rxh5 Re3 Rh6 Ke7 h5 e5 Kd5 f3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 a3 Be6 Qb3 Qd7 Nxe6 fxe6 Rad1 Bd6 Bc1 Kh8 Qa4 Qe7 e3 a6 Qh4 Rac8 e4 d4 Ne2 e5 Bh3 Rc7 Bg5 Kg8 Bxf6 Qxf6 Qxf6 gxf6 Nc1 Na5 Nd3 Nb3 Bf5 a5 Kg2 Kg7 Kh3 Ree7 Nc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 Qa4 Nd7 Nxd5 exd5 Bf4 c6 g3 Be7 Bh3 O-O Rc1 Bf6 O-O Re8 Rxc6 Bxc6 Qxc6 Nf8 e3 Ne6 Bd6 Be7 Bxe7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bd2 Be4 Qc1 Nc6 Be3 Nb4 Nbd2 Bb7 Bg5 Rc8 a3 Nbd5 Nb3 h6 Na5 Ba8 Nc6 Bxc6 Bxf6 Bb7 Bxe7 Qxe7 Qc5 Qxc5 dxc5 Ne7 a4 b4 Nd4 Bxg2 Kxg2 Rfd8 Rfd1 Rd5 Nc2 Rb8 Rxd5 Nxd5 Nd4 Ne7 Rd1 Kf8 Nb3 Nc6 f4 Ke7 Kf3 g6 Rd2 f6 Ke4 f5 Kd3 e5 e4 Ke6 Ke3 exf4 gxf4 g5 Nd4 Nxd4 Rxd4 gxf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 c5 O-O Nbd7 Na3 Nb6 Nxc4 Nxc4 Qa4 Bd7 Qxc4 b5 Qc2 Rc8 dxc5 Bxc5 Qb3 O-O Ne5 Qb6 Bg5 Rfd8 Qf3 Be7 Nxd7 Rxd7 Rac1 Rcd8 Qc6 Qa5 a3 b4 Bf4 Nd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 O-O Be7 Be3 c4 Ne5 O-O b3 cxb3 Qxb3 Qb6 Rc1 Qxb3 axb3 Nb4 Na3 a6 Bd2 Rb8 Bxb4 Bxb4 Nd3 Bd6 Nc2 Bg4 Kf1 Bf5 Nc5 Rfc8 Ne3 Be6 b4 Kf8 Rc2 Ke7 Ke1 h5 Rb2 Rc7 Nd3 Ra8 b5 a5 b6 Rc6 Rb5 a4 Nxd5 Nxd5 Bxd5 Bxd5 Rxd5 Rxb6 Rxh5 Rb3 Kd2 b5 h4 Rc8 g4 a3 f4 Rcc3 Rd5 Ke6 Rh5 b4 Ra5 Rxd3 exd3 Bxf4 Ke2 Rc3 g5 Bc1 h5 b3 R5xa3 Bxa3 Rxa3 b2 Ra6 Kf5 Rb6 Rc2 Ke3 Kxg5 d5 Kxh5 Kd4 g5 Rb8 g4 d6 Rc6 Ke5 Rc5 Kf6 g3 Rxb2 Rd5 Kxf7 Rxd6 Rd2 Kg4 d4 Kf5 Ke7 Rd5 Rd3 Kf4 Ke6 Rg5 d5 Rg6 Ke7 g2 Rd1 Ke5 d6 Re6 Kd7 Rxd6 Rxd6 g1=Q Re6 Kf5 Rd6 Qa7 Kd8 Ke5 Rg6 Qa5 Kd7 Qa4 Ke7 Qh4 Kf8 Qd8 Kf7 Kf5 Rh6 Qd7 Kf8 Kg5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 Bd7 Qc2 c5 O-O Bc6 Qxc4 Nbd7 Bg5 Rc8 Bxf6 Nxf6 dxc5 Bxf3 Bxf3 Bxc5 Qb5 Qd7 Nc3 Qxb5 Nxb5 Ke7 b4 Bxb4 Nxa7 Rc7 Rfc1 Rd7 Rab1 Bd2 Rc2 Rhd8 Bxb7 Kf8 Nc6 Rc7 Rbb2 Rd6 a4 Be1 Rb1 Nd5 Ba8 Rc8 Bb7 Rc7 Rc4 Ne7 Ne5 Ba5 Rb5 Ng6 Nc6 Rd1 Kg2 Be1 a5 Ne7 a6 Nxc6 Rxc6 Rxc6 Bxc6 Ra1 Rb8 Ke7 Rb7 Kd6 Bb5 Bc3 Rxf7 Bf6 Rd7 Kc5 Bd3 h6 Rb7 Ra3 a7 Kd5 f3 Kd6 Rb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 g3 dxc4 Bg2 c5 Qa4 Bd7 Qxc4 Bc6 dxc5 Nbd7 Be3 Bd5 Qa4 Bc6 Qc4 Bd5 Qb4 Qc8 Nc3 Bxc5 Bxc5 Qxc5 Nxd5 Nxd5 Qd2 Rc8 O-O O-O Rac1 Qb6 Qd4 Rfd8 Rfd1 Qxd4 Nxd4 N7b6 Nb3 Rxc1 Rxc1 Rc8 Rxc8 Nxc8 Bxd5 exd5 Nc5 Nd6 Kg2 Kf8 Kf3 Ke7 Kf4 f6 h4 g6 g4 b6 Na6 Ne4 f3 Nc5 Nc7 d4 Nd5 Ke6 Nb4 a5 Nd3 Kd5 g5 f5 Kg3 Nxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 Nbd7 O-O Rb8 a4 b6 Nfd2 e5 Nxc4 exd4 Qxd4 Bc5 Qd3 O-O Nc3 Bb7 Bxb7 Rxb7 Qf3 Qa8 Bf4 a6 e4 Ra7 Nd5 b5 Na5 bxa4 Rfc1 Bd4 Rxa4 Bxb2 Ne7 Kh8 Rc2 Qe8 Rxb2 Qxe7 Nc6 Qc5 Nxa7 Qxa7 e5 Ng8 Be3 Qa8 Qxa8 Rxa8 f4 Ne7 Rd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 Bg5 Be7 Nc3 h6 Bh4 O-O e3 b6 Qb3 Bb7 Bxf6 Bxf6 cxd5 exd5 Rd1 Re8 a3 c6 Bd3 Nd7 O-O g6 Rfe1 Nf8 Bb1 Ne6 Ba2 Qc7 Qa4 Rad8 b4 Qb8 Qc2 Qc7 Bb3 Bg7 Qa2 a6 Rc1 Qb8 Na4 Qa7 Nc3 Qb8 Rb1 Qd6 Rbd1 a5 bxa5 bxa5 e4 a4 Bxa4 dxe4 Nxe4 Qf4 d5 Nd4 Nxd4 Rxe4 Bxc6 Bxd4 Bxb7 Rb8 Rxe4 Qxe4 Qb1 Qf4 d6 Qxd6 g3 Rxb7 Qxb7 Bxf2 Kxf2 Qxd1 Qa8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 c5 d5 exd5 cxd5 b5 Bg2 d6 b4 Na6 bxc5 Nxc5 Nf3 g6 O-O Bg7 Nd4 O-O Nc3 a6 Nc6 Qc7 Be3 Bb7 Bd4 Rfe8 a4 bxa4 Bxc5 dxc5 Qxa4 Nd7 Qb3 Bxc6 dxc6 Nb6 Rab1 Rab8 Qa3 c4 Rfc1 Bxc3 Qxc3 Rxe2 Qd4 a5 Rb5 a4 Bf3 Ree8 Qc5 Qe7 c7 Qxc5 Rxc5 Rbc8 Bb7 Nd7 R5xc4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 c6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Bb7 g3 c5 d5 Qb6 Bg2 O-O-O O-O b4 Na4 Qb5 a3 Nb8 axb4 cxb4 Be3 Bxd5 Bxd5 Rxd5 Qe2 Nc6 Rfc1 Ne5 b3 c3 Nxc3 bxc3 Rxc3 Kb8 Qc2 Bd6 Bxa7 Kb7 b4 Nc6 Be3 Be5 Rxc6 Bxa1 Rc7 Kb8 Ba7 Ka8 Be3 Kb8 Ba7 Ka8 Bc5 Kb8 Rxf7 Be5 Ba7 Ka8 Be3 Rd7 Qa2 Kb8 Ba7 Kc8 Qxe6 Qd5 Qa6 Qb7 Qc4 Qc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 cxd5 exd5 Bg5 Be7 e3 h6 Bh4 O-O Bd3 b6 Nf3 Bb7 O-O c5 Ne5 Nc6 Ba6 Qc8 Bxb7 Qxb7 Bxf6 Bxf6 Ng4 Bd8 Nxd5 Nxd4 Ndf6 Bxf6 Nxf6 gxf6 exd4 cxd4 Qxd4 Kg7 Rac1 Rac8 Qg4 Kh7 Qf4 Rg8 Qf5 Kg7 h4 Rge8 Qg4 Kh7 Qf4 Kg7 Rxc8 Qxc8 Rd1 Qe6 Rd3 Qe1 Kh2 Qe5 Rg3 Kh7 Qc4 Qe6 Qd4 Qf5 Qc4 Qe6 Qc7 Qe7 Qc6 Qe6 Qb7 Qe7 Qd5 Qe6 Qh5 Rd8 Re3 Rd5 Qf3 Qf5 Qxf5 Rxf5 Kg3 Kg6 Re7 Ra5 a3 Rb5 b4 a5 Re4 Rd5 f3 h5 Kf4 Rd3 a4 f5 Rc4 axb4 Rxb4 f6 Kg3 Rd6 Kf2 Re6 g3 Kg7 Rc4 Kg6 Rc8 Re5 Ra8 Kg7 Ra6 b5 f4 Rc5 Ra7 Kg6 a5 b4 a6 b3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8 Qa4 Bd7 Rad1 Nb4 Qb3 a5 Rd2 a4 Qd1 a3 Qb1 Bf8 bxa3 Rxa3 Qb2 Qa8 Nb3 Bc6 Bd4 Ne4 Nxe4 dxe4 Ra1 Bd5 Qb1 b6 e3 Nd3 Rd1 b5 Bf1 b4 Bxd3 exd3 Qxd3 Rxa2 Rxa2 Qxa2 Nc5 Bf3 Ra1 Qd5 Qb3 Qh5 Nd3 Bd6 Ne1 Bb7 Rc1 Qf5 Rd1 Bf8 Qb1
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 h6 Bh4 O-O Bd3 b6 Nf3 Bb7 O-O c5 Ne5 Nbd7 Qf3 cxd4 exd4 Nxe5 dxe5 Nd7 Bxe7 Qxe7 Nxd5 Qxe5 Ne7 Kh8 Qxb7 Nc5 Qf3 Nxd3 Nc6 Qe6 b3 Ne5 Nxe5 Qxe5 Rae1 Qc7 Rc1 Qe7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 c5 Nf3 d5 O-O dxc4 Bxc4 cxd4 exd4 b6 Qe2 Bb7 Rd1 Bxc3 bxc3 Qc7 Bd3 Qxc3 Bb2 Qc7 d5 Bxd5 Bxf6 gxf6 Qe3 Kg7 Rac1 Nc6 Be4 Qd6 Bxd5 exd5 Rc4 Qd7 Rh4 Qf5 Rxd5 Ne5 h3 Rfe8 Nd4 Qg6 Qf4 Rad8 Nf5 Kh8 Rxd8 Rxd8 Qe4 Rc8 Kh2 Rc4 Qa8 Qg8 Qxa7 Rxh4 Nxh4 Qg5 Qa8 Kg7 Qe4 h5 Nf5 Kg6 Ne7 Kh6 f4
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 h6 Bh4 O-O Bd3 b6 Nf3 Bb7 O-O c5 Ne5 Nbd7 Bf5 Nxe5 dxe5 Ne8 Bg3 Nc7 Qg4 Qe8 Bd7 Qd8 Rad1 h5 Qh3 h4 Bf4 Bg5 Bf5 g6 Ne4 Bxf4 exf4 gxf5 Qxf5 dxe4 Qg4 Kh7 Rxd8 Rfxd8 Qxh4 Kg8 Qe7 e3 Re1 exf2 Kxf2 Rd2 Re2 Rxe2 Kxe2 Ba6 Kf2 Ne6 f5 Nd4 e6 Rf8 Qg5 Kh7 e7 Re8 f6 Ne6 Qh5 Kg8
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8 Qc2 Bg4 Rfd1 Bf8 Rac1 Rc8 Nxc6 bxc6 Bd4 Bb4 Rd2 Qe7 a3 Ba5 b4 Bb6 e3 Qe6 Qb2 Bxd4 Rxd4 c5 bxc5 Rxc5 Ne2 Rec8 Rxc5 Rxc5 Nf4 Qc8 h3 Rc1 Kh2 Rc2 Qb3 Bf5 Kg1 Rc1 Kh2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 h6 Bh4 O-O Bd3 b6 Nf3 Bb7 O-O Ne4 Bxe7 Qxe7 Ne5 Nd7 f4 Nxe5 fxe5 c5 Qe1 Rad8 Rd1 Qg5 Rf3 f6 exf6 cxd4 exd4 Rde8 Bb5 Rd8 Bd3 Rde8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 a6 Bd3 c5 dxc5 dxc5 Bxc5 Nc6 Nge2 Nd7 Bf2 Nde5 Nc1 Bh6 Nd5 e6 Bb6 Qg5 O-O exd5 f4 Qh4 fxe5 d4 Ne2 Be3 Kh1 Nxe5 Bc7 Qe7 Bxe5 Qxe5 Qe1 Bd7 Qg3 Rae8 Nf4 Bc6 Nd5 Qxg3 hxg3 Re5 g4 h5 Nf6 Kg7 gxh5 Rh8 g3 Rexh5 Nxh5 Rxh5 Kg2 f5 Rae1 fxe4 Bb1 Rc5 b3 b5 Rxe3 dxe3 Re1 bxc4 bxc4 Rxc4 Rxe3 Rb4 Rb3 e3 Kf1 Bb5 Ke1 a5 Be4 Rxb3 axb3 Kf6 Kd1 g5 Kc2 Ke5 Bg6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c5 d5 d6 Nc3 g6 e4 Bg7 Bb5 Bd7 a4 O-O O-O Na6 Re1 Nb4 h3 e6 Bf4 e5 Bg5 Bc8 Nd2 h6 Bh4 g5 Bg3 g4 hxg4 Nxg4 f3 Nf6 Bh4 Kh8 Ne2 Rg8 c3 Na6 Ng3 Qf8 Ndf1 Nh7 Ne3 Bf6 Bxf6 Nxf6 Ngf5 Nh5 Kf2 Bxf5 Nxf5 Nf4 g3 Nh3 Ke2 Rxg3 Nxg3 Qg7 Rg1 Rg8 Qd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Bxf6 Bxf6 Nb3 d4 Ne4 Be7 Rc1 Qb6 Nec5 Rd8 Rc4 Bxc5 Nxc5 Qxb2 Qc2 Qxc2 Rxc2 Rb8 Rb2 Rd6 Rd1 b6 Nb3 Bb7 Rbd2 Rbd8 Kf1 Ba6 Bxc6 Rxc6 Nxd4 Rc5 Nb3 Rxd2 Rxd2 Rc7 Rd8 Kh7 Ke1 Bc4 Kd2 g6 Nc1 Kg7 a3 Kf6 e3 Ke7 Rd4 Rd7 Kc3 Rxd4 Kxd4 b5 Nd3 Bxd3 Kxd3 Kd6 e4 g5 f4 gxf4 gxf4 Kc5 Kc3 a5 Kd3 h5 h4 b4 a4 f6 f5 Kc6 Kc4 Kc7 Kd3 Kd7 Ke3 Kc6 Kd3 Kc5 Ke3 b3 Kd3 Kb4 e5 Ka3
Result: 0-1
Processing game...
FEN: 
Moves: d4 e6 c4 f5 g3 Nf6 Bg2 d5 Nf3 Be7 O-O O-O b3 c6 Qc2 Bd7 Bb2 Be8 Ne5 Nbd7 Nd3 Bh5 Nc3 Bd6 f3 Bg6 e3 Rc8 Qe2 Re8 Qf2 a6 Rac1 Qe7 Rfe1 Qf8 Rcd1 dxc4 bxc4 c5 Bf1 Bf7 Na4 cxd4 exd4 b5 cxb5 axb5 Nac5 b4 Rc1 Qe7 Bh3 Qd8 Nxb4 Qa5 Nc6 Qxa2 Nxd7 Nxd7 d5 Qxd5 Red1 Bc5 Rxd5 Bxf2 Kxf2 exd5 Bxf5 Nb6 Bxc8 Nxc8 Ba3 h6 Rb1 Re6 Nd4 Ra6 Bc5 Nd6 Rb8 Kh7 g4 Ra4 Ke3 Nc4 Kf4 g5 Kg3 Ra2 Rb7 Kg6 Nf5 Ra6 h4 gxh4 Nxh4 Kg7 Nf5 Kg6 Bd4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be3 Nf6 f4 Be7 Qf3 O-O O-O-O Qc7 g4 Nxd4 Bxd4 e5 fxe5 dxe5 Qg3 Nxg4 Nd5 Qd8 Nxe7 Qxe7 Bc3 Qc5 Rg1 Qe3 Kb1 Be6 h3 Rfd8 Re1 Qxg3 Rxg3 Nf6 Bxe5 Ne8 Bb5 f6 Bc3 Rac8 Bxe8 Rxe8 Bxf6 Rc7 Reg1 g6 h4 Rc4 h5 Kf7 Bc3 Rg8 hxg6 Rxg6 Rxg6 hxg6 Rh1 Bd7 e5 Rf4 Kc1 Bc6 Rh7 Ke6 Rg7 Be4 Kd2 Rf7 Rxf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 e3 Nxc3 bxc3 Be7 Bb5 c6 Bd3 c5 O-O Nc6 Bb2 Rc8 Qe2 O-O Rad1 Qc7 c4 cxd4 exd4 Na5 d5 exd5 cxd5 Bxd5 Bxh7 Kxh7 Rxd5 Kg8 Bxg7 Kxg7 Ne5 Rfd8 Qg4 Kf8 Qf5 f6 Nd7 Rxd7 Rxd7 Qc5 Qh7 Rc7 Qh8 Kf7 Rd3 Nc4 Rfd1 Ne5 Qh7 Ke6 Qg8 Kf5 g4 Kf4 Rd4 Kf3 Qb3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d3 Nc6 g3 d5 Nbd2 g6 Bg2 Bg7 O-O Nge7 Re1 b6 c3 h6 h4 a5 a4 Ra7 Nb3 d4 cxd4 cxd4 Bd2 e5 Nc1 Be6 Re2 O-O Be1 f5 Nd2 f4 f3 fxg3 Bxg3 g5 hxg5 Ng6 gxh6 Bxh6 Nf1 Rg7 Rf2 Be3 b3 Nf4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 a3 Bxc3 bxc3 O-O Bd3 d5 cxd5 exd5 Ne2 b6 f3 Re8 O-O Ba6 Ng3 Bxd3 Qxd3 Nc6 Bb2 c4 Qd2 Qd7 Rae1 h5 e4 g6 Bc1 Nh7 Qh6 Re6 f4 Ne7 f5 gxf5 Qxh5 dxe4 Nxf5 Nxf5 Rxf5 Rae8 Re3 Rg6 Rh3
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O b3 Ne4 Bb2 Bf6 Na4 Re8 Rc1 b6 dxc5 Bxb2 Nxb2 bxc5 Na4 Ba6 Re1 c4 Nh4 Qa5 Nf5 g6 Nd4 Rac8 h4 Ne5 Bh3 Rc7 Nc2 cxb3 axb3 Bc8 Bg2 Ng4 Rf1 Bd7 Ra1 Bxa4 Rxa4 Qc3 Bxe4 dxe4 e3 Qxb3 Rxe4 Rxe4 Qd8 Kg7 Qxc7 Rc4 Nd4 Rxc7 Nxb3 Rc2 Nd4 Ra2 e4 Rd2 Nc6 a6 e5 Re2 Ra1 Rxf2 Rxa6 Rc2 h5 Kh6 hxg6 hxg6 Ra4 Kg5 Nd4 Rc3 e6 Rxg3 Kh1 f5 e7 Re3 Nc6 f4 Ra5 Kh4 Ra8 Nf6 Kg2 f3 Kf1 Kg3 Nd4 Ng4 Nxf3 Rxf3 Kg1 Nh2 Rf8 Rc3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 Nc6 Be3 a6 Nge2 Rb8 Qd2 O-O h4 b5 h5 bxc4 g4 Bxg4 fxg4 Nxg4 O-O-O Nxe3 Qxe3 e6 hxg6 hxg6 Rd2 Re8 Ng1 d5 Nf3 a5 e5 Ne7 Bh3 c5 dxc5 Qc7 Qf4 Nc6 Re1 d4 Rxd4 Nxd4 Nxd4 Qxc5 Nf3 Red8 Ng5 Qe7 Qh4 Rd3 Qh7 Kf8 Nxe6 fxe6 Rf1 Ke8 Qg8 Bf8 Qxg6 Kd8
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O dxc5 Bxc5 Bg5 d4 Ne4 Be7 Bxf6 Bxf6 Rc1 Re8 Ne1 Be7 Nd3 Bf8 Qd2 a5 Rfd1 Bg4 Ndc5 Bxc5 Rxc5 Qe7 h3 Bxe2 Re1 d3 Qc3 Rad8 Nd2 Nd4 Qxa5 h6 Rc3 b6 Qa6 Qg5 Rxd3 Qg6 Bf1 Qxd3 Qxd3 Bxd3 Rxe8 Rxe8 Bxd3 Re1 Bf1 Ra1 Nc4 b5 Nd6 Rxa2 Nxb5 Nxb5 Bxb5 Rxb2 Bc4 Rc2 Bd5 Kf8 h4 g6 Kg2 Ke7 Kf3 Rc7 Ke4 Kd6 Ba2 Re7 Kd4 Ra7 Bb3 Ke7 Ke4 Kf6 Bd5 Re7 Kf4 Re2 Kf3 Rd2 Bc4 Rd4 Ba2 Rd7 Bc4 Ke5 Ba2 Kd4 Bb1 Rc7 Ba2 Ra7 Bb1 Ra1 Bc2 Kc3 Be4 Kd2 Kg2 Ke1 Bd5 Ra7 f4 Ke2 h5 gxh5 Bf3 Ke3 Bxh5 Ra2 Kh3 f6 Be8 Kf2 Kh4 Ra8 Bc6 Rg8 g4 Ke3 Kg3 f5 Bf3 h5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 a5 Nc3 b6 e3 Bb7 Bd3 d6 Qc2 Nbd7 e4 e5 Nd5 Bxd2 Qxd2 exd4 Nxd4 Nc5 O-O O-O Rfe1 Re8 f3 c6 Nc3 Qc7 Bf1 Rad8 Rad1 d5 cxd5 cxd5 Ncb5 Qb8 e5 Rxe5 Rxe5 Qxe5 Re1 Qb8 Nf5 Ne6 Nbd4 Re8 Bb5 Nxd4 Rxe8 Nxe8 Qxd4 Qc7 Bxe8 Qc1 Kf2 Qc2 Ke3 Qxf5 Qxb6 Qg5 Kd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 O-O Be7 Nc3 O-O dxc5 Bxc5 Bg5 d4 Bxf6 Qxf6 Nd5 Qd8 Nd2 Re8 Rc1 Bb6 Re1 Be6 Nf4 Qd7 Qa4 Rac8 Nc4 Bd8 Nxe6 Qxe6 Qb5 Rc7 Qd5 Qf6 Qf3 Qe6 Qd5 Qf6 Qd6 Re6 Qa3 Rce7 Qd3 Re8 a3 h5 Bd5 R6e7 h4 Rd7 Bf3 g5 hxg5 Qxg5 Kg2 Ne7 Qd2 Ng6 Qxg5 Bxg5 Rcd1 h4 Bg4 Rdd8 f4 b5 Na5 Re3 Bf3 Bf6 Nc6 Rde8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 c5 O-O Nc6 Qa4 cxd4 Nxd4 Qxd4 Bxc6 Bd7 Rd1 Qxd1 Qxd1 Bxc6 Nd2 b5 a4 Be7 axb5 Bxb5 Nxc4 O-O b3 Rfd8 Qc2 Rdc8 Ba3 Bxa3 Rxa3 h6 Qc3 Bxc4 bxc4 Rc7 Qd4 Rac8 Ra4 Rd7 Qa1 Rdc7 Qa2 Ne4 Rxa7 Rxc4 Qb2 Nf6 Qb7 Rf8 f3 Rd4 Qe7 Rdd8 Kf2 Ra8 e4 Rxa7 Qxa7 e5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Bxf6 gxf6 Be2 Nc6 Qd2 Bd7 O-O-O h5 Kb1 Be7 Bf3 Nxd4 Qxd4 O-O-O f5 Kb8 Qd2 h4 Ne2 Bc8 fxe6 fxe6 Nf4 Rhg8 Qe3 Rg5 Rd3 Bf8 Rhd1 Bh6 Ne2 h3 g3 Rh5 Qf2 Rc5 Nd4 Rc4 Qe2 Rg8 b3 Rc5 Qf2 Re5 Qf1 d5 Qxh3 dxe4 Bxe4 Bg7 Bf3 f5 b4 Re4 c3 e5 Nb3 Rc4 Bd5 Rh8 Qg2 Rxc3 Rxc3 Qxc3 Qd2 Qc7 Na5 e4 Nc4 Rd8 Qf4 Qxf4 gxf4 Ka7 Ne3 Bh6 Rf1 Bf8 a3 a5 h4 axb4 axb4 Bxb4 Bf7 Rd3 Nd5 Ba5 Rf2 Bd7 Ra2 b6 Rc2 Ba4 Rc7 Kb8 Re7 Bb3 Re5 e3 h5 Bxd5 Bxd5 Rxd5 Rxe3 Rd1 Kc2 Rh1 Re8 Kb7 Rh8 b5 h6 Rh2 Kd3 Rh3 Kc2 b4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 b6 g3 Bb7 c4 e6 Bg2 Be7 Nc3 Ne4 Bd2 d5 cxd5 exd5 Qa4 Qd7 Qc2 Nxd2 Qxd2 O-O O-O Nc6 Rac1 Rad8 e3 Bf6 Rfd1 Na5 b3 Rfe8 Ne2 Qd6 Nf4 c6 Ne5 c5 Ng4 Be7 dxc5 Qd7 Ne5 Qc7 cxb6 Qxb6 Nxd5 Ba3 Nxb6 Rxd2 Nbc4 Nxc4 Nxc4 Bxg2 Nxd2 Bh3 Rc4 a5 Re4 Ra8 Nc4 Bf8 Rd5 g6 Rxa5 Rc8 f3 h5 Rd4 Bg7 Rd3 Be6 Rad5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Bg5 Be7 e3 O-O cxd5 exd5 Nc3 Nbd7 Bd3 c6 Qc2 Re8 O-O Nf8 Bxf6 Bxf6 b4 Bg4 Nd2 Be7 Rab1 Bd6 Bf5 Bh5 Rfc1 g6 Bd3 Qg5 Ne2 Nd7 h3 a6 a4 Rac8 Nf1 Bxe2 Bxe2 Qe7 Qb3 Nf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Rc1 Bb7 cxd5 Nxd5 Nxd5 Bxd5 Bxe7 Qxe7 Be2 Rc8 O-O c5 dxc5 Rxc5 Rxc5 Qxc5 Qa4 Bc6 Qf4 Nd7 b4 Qf8 Qc7 Rc8 Qxa7 Qxb4 Nd4 Qa4 Nxc6 Qxc6 Rd1 Ne5 Qe7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Re8 d4 Bb7 Ng5 Rf8 Nf3 Re8 Nbd2 Bf8 Bc2 Nb8 b4 Nbd7 a4 Nb6 axb5 axb5 Rxa8 Qxa8 Bd3 exd4 Nxd4 Nxe4 Nxe4 Bxe4 Bxe4 Rxe4 Nxb5 Qd5 Nd4 g6 Rxe4 Qxe4 Qe2 Qb1 Qe1 Nc4 Qf1 Qa1 Qe1 Qb1 Kf1 h6 Bf4 Qd3 Kg1 g5 Bc1 d5 Be3 Bg7 g4 Kh7 Qc1 Be5 Kg2 Kg6 Kg1 Bxd4 Bxd4 Nd2 Kg2 Qe4 Kg3 Qd3 Kg2 Qe4 Kg3 Qe2 Kg2 f5 gxf5 Kf7 Be3 Nf3 Qh1 Qd3 Kg3 Nh4 Kh2 Nxf5 Bd4 c6 Qa1 Nxd4 cxd4 Qd2 Qa7 Kg6 Kg2 Qxb4 Qd7 Qxd4 Qe8 Kf5 Qd7 Kf4 Qf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bxf6 Bxf6 Qc2 c5 dxc5 Qa5 cxd5 exd5 O-O-O Be6 Nxd5 Rc8 Kb1 Bxd5 Rxd5 Nc6 Bc4 Nb4 Qd2 Rxc5 Rxc5 Qxc5 Rc1 Qb6 Qd7 Rf8 Qb5 Qd6 e4 Nc6 Bd5 a6 Qxb7 Ne5 Rc8 Rxc8 Qxc8 Kh7 Qc2 Kg8 Nd2 g5 a3 Kg7 Nf1 Qb6 Ng3 Kg6 Ka2 h5 Qc8 h4 Qg8 Bg7 Nh5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 g4 h6 h4 Nc6 Rg1 h5 gxh5 Nxh5 Bg5 Nf6 Qd2 Qb6 Nb3 Bd7 O-O-O a6 Rg3 Qc7 Bg2 Be7 f4 O-O-O Qf2 Kb8 f5 Ne5 Bh3 Nc4 Nd2 Nxd2 Rxd2 Rc8 fxe6 Bxe6 Bxe6 fxe6 Qg1 Qa5 Qd4 Qc5 Qd3 Qc4 Qe3 Ka8 a3 Qc6 e5 dxe5 Qxe5 Rhd8 Rgd3 Rxd3 Rxd3 Qh1 Nd1 Qg2 Rd2 Qc6 Re2 Bd6 Qc3 Qd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O d5 exd5 Nh4 c6 cxd5 Nxd5 Nf5 Nc7 Nc3 d5 e4 Bf6 Bf4 Bc8 g4 Nba6 Rc1 Bd7 Qd2 Nc5 e5 Be7 Nxe7 Qxe7 Bg5 Qe6 h3 Qg6 f4 f6 exf6 gxf6 Bh4 f5 b4 fxg4 hxg4 Nd3 Rf3 Nxc1 f5 Qg7 Qxc1 Rae8 Qd2 d4 Ne2 Nd5 Nxd4 Kh8 g5 Re4 Bf2 Qe5 Rg3 Rf4 f6 Be8 b5 c5 Nc6 Qa1 Bf1 Rf5 g6 Bxg6 Rxg6 R5xf6 Rxf6 Qxf6 Qe1 Rg8 Kh2 Qf4 Bg3 Rxg3 Qxg3 Qxf1 Qb8 Kg7 Qg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 Be7 Be2 O-O O-O b6 Be3 Bb7 Qb3 Na5 Qxb6 Nxe4 Nxe4 Bxe4 Qxd8 Bxd8 Rad1 d5 f3 Bf5 cxd5 exd5 Rxd5 Be6 Rd6 Bxa2 Rxa6 Rb8 Bc5 Re8 Bb5 Re6 b4 Nb7 Bf2 Be7 Nc2 Bd5 Rd1 Bb3 Rd7 Rd8 Rxe6 Rxd7 Re1 Rc7 Bb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 Bb7 Nc3 d5 cxd5 exd5 O-O O-O Bf4 Na6 Qc2 c5 Rfd1 Qc8 Be5 Rd8 Rac1 Ne4 Qb2 Qe6 Nb5 Bf8 Bf4 Qe8 a4 Bc6 dxc5 bxc5 Ne5 Bxb5 axb5 Nb4 Qb1 Nf6 Nc6 Nxc6 bxc6 Qxc6 Bg5 a5 Bxf6 Qxf6 Bxd5 Ra7 Rc4 Qb6 Qc2 Rad7 e4 Kh8 Kg2 f5 f3 g6 Rc3 Qc7 Rcd3 fxe4 fxe4 Bg7 Rc1 Rc8 Rf3 Qe5 Rcf1 Qd6 Qe2 Ra7 Qe3 Qe5 R1f2 a4 bxa4 Rxa4 Rf7 Rb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 Bf3 Nc6 a4 Re8 Be3 Rb8 Re1 Bd7 Qd3 Nxd4 Bxd4 e5 Ba7 Rbc8 Be3 Qc4 a5 h6 h3 Bf8 Bd2 Qd4 Be3 Qb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 O-O O-O d5 Ne5 c6 Bc3 Nfd7 Nxd7 Nxd7 Nd2 Rc8 e4 b5 Re1 dxc4 bxc4 Nb6 cxb5 cxb5 Rc1 Ba3 Rc2 Na4 Ba1 Rxc2 Qxc2 Qa5 Qd1 Rc8 Nb3 Qb4 d5 exd5 exd5 Nc3 Qd4 Qxd4 Nxd4 Nxa2 Nc6 Bc5 Bh3 Ra8 Bd4 Bxd4 Nxd4 Kf8 d6 Nc3 Nc6 Bb7 Bg2 Re8 Ne5 f6 d7 Rd8 Bxb7 fxe5 Bc6 Ke7 Bxb5 Nxb5 Rxe5 Kxd7 Rxb5 Kc6 Rh5 h6 Re5 Ra8 Ra5 Kb6 Ra2 a5 Kf1 a4 Ke2 Kc5 Kd2 a3 Kc1 Kd4 f4 Ke4 Kb1 Rb8 Ka1 Rb2 Rxa3 Rxh2 Kb1 Rd2 Ra6 Kf5 Ra7 g5 Ra6 g4 Rxh6 Rg2 Rh5 Ke4 f5 Rf2 Kc1 Kf3 Kd1 Kxg3 Ke1 Kg2 Rg5 g3 Rh5 Rf4 Ke2 Re4 Kd3 Kf3 Rh1 g2 Rh3 Kg4 Rh8 Rf4 Ke2 Rxf5
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 Qb3 Na5 Qc2 Bg4 Nf5 Rc8 Nxe7 Rxe7 Rad1 Qe8 h3 Bh5 Bxd5 Bg6 Qc1 Nxd5 Rxd5 Nc4 Bd4 Rec7 b3 Nb6 Re5 Qd7 Qe3 f6 Rc5 Rxc5 Bxc5 Qxh3 Rd1 h5 Rd4 Nd7 Bd6 Bf7 Nd5 Bxd5 Rxd5 a6 Bf4 Nf8 Qd3 Qg4 f3 Qg6 Kf2 Rc2 Qe3 Rc8 Qe7 b5 Rd8 Rxd8 Qxd8 Qf7 Bd6 g5 Qa8 Kg7 Qxa6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bd2 Be4 Qc1 Bb7 Be3 Nd5 Nc3 Nd7 Rd1 Rc8 Nxd5 Bxd5 Ne1 c6 Nd3 Qb6 Qc3 b4 Qd2 a5 Rdc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 Qb3 Na5 Qc2 Bg4 Nf5 Rc8 Bd4 Bc5 Bxc5 Rxc5 Ne3 Be6 Rad1 Qc8 Qa4 Rd8 Rd3 a6 Rfd1 Nc4 Nxc4 Rxc4 Qa5 Rc5 Qb6 Rd7 Rd4 Qc7 Qxc7 Rdxc7 h3 h5 a3 g6 e3 Kg7 Kh2 Rc4 Bf3 b5 Kg2 R7c5 Rxc4 Rxc4 Rd4 Kf8 Be2 Rxd4 exd4 Ke7 Na2 Bc8 Nb4 Kd6 f3 Ng8 h4 Nh6 Kf2 Nf5 Nc2 f6 Bd3 g5 Bxf5 Bxf5 Ne3 Bb1 b4 gxh4 Ng2 hxg3 Kxg3 Ke6 Nf4 Kf5 Nxh5 Ke6 Nf4 Kd6 Kg4 Bc2 Kh5 Bd1 Kg6 Ke7 Nxd5 Ke6 Nc7 Kd7 Nxa6 Bxf3 Kxf6 Kd6 Kf5 Kd5 Kf4 Bh1 Ke3 Kc4 Nc5 Bc6 Nd3 Bg2 Ne5 Kc3 Ng6 Kc4 Ne7 Bb7 Nf5 Bg2 Nd6 Kb3 Nxb5 Ka4 Nd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 e3 Nd7 Bd3 c5 e4 N5f6 d5 exd5 exd5 Bd6 O-O O-O Bg5 Qc7 Bf5 a6 Qd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 b3 Bg7 Bb2 O-O e3 e6 d4 Qe7 Nc3 Na6 Qe2 d5 Rfd1 Rfd8 Rac1 dxc4 bxc4 Rac8 Nb5 Be4 a3 Nb8 dxc5 bxc5 Be5 Ne8 Bxg7 Kxg7 Ne5 Bxg2 Kxg2 f6 Nd3 Nc6 Nc3 Nd6 Na4 Ne5 Nxe5 fxe5 Nc3 Qb7 Qf3 Qxf3 Kxf3 Rb8 Rb1 e4 Ke2 Rxb1 Rxb1 Kf6 Rd1 Ke5 Nb5 Rd7 g4 h6 a4 a6 Nxd6 Rxd6 Rb1 Rd3 f4 Kf6 Rb6 Rc3 Rxa6 Rc2 Kd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qd2 Rfd8 Rfc1 Rab8 Bxc6 Rxb1 Nxb1 Qxc6 dxc5 Nxc5 Qc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 Nc3 Bg7 d4 cxd4 Nxd4 Bxg2 Kxg2 O-O e4 Qc7 b3 Nxe4 Nxe4 Qe5 Qf3 Qxd4 Ba3 Nc6 Rad1 Qe5 Rxd7 Qa5 Bxe7 Ne5 Qd1 Nxd7 Qxd7 Qxa2 Bxf8 Rxf8 Re1 Qxb3 Nd6 Qc3 Re7 Qf6 Ne4 Qd4 Qxd4 Bxd4 Rd7 Bg7 Rxa7 h6 Rb7 Bd4 Rd7 Bg7 h4 f5 Nd2 Rf6 Rc7 Re6 Nf3 Bf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 O-O O-O d5 cxd5 Nxd5 Nc3 Nd7 Nxd5 exd5 Rc1 c5 dxc5 bxc5 Ne1 Nb6 a4 Rc8 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 d4 e6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 O-O O-O d5 cxd5 Nxd5 Nc3 Nd7 Nxd5 exd5 Rc1 Re8 Rc2 c5 Re1 Bb7 Be3 a5 Qc1 a4 Rd1 axb3 axb3 Bf6 Ne1 h6 Bf3 Qe7 Qd2 Ra3 Rb1 Rd8 dxc5 Nxc5 Qb4 d4 Bxd4 Rxb3 Rxb3 Rxd4 Qxb6 Nxb3 Rc7 Rd7 Rxb7 Rxb7 Qxb7 Nd4 Kf1 Qxb7 Bxb7 Nf5 Nd3 Bc3 e3 g5 g4 Nd6 Bf3 Kg7 Ke2 Kf6 h3 Ke7 Bd5 Kf6 Nc5 Ke7 Na4 Ba5 Nc5 Bb6 Na6 Ba5 Nb8 Bc3 Nc6 Kd7 Kd3 Be1 Ne5 Ke7 Ke2 Bc3 Nf3 Ba5 Bb3 Bb6 Ne5 Bc5 Kd3 Bb6 Bd5 Bc5 Kc3 Ke8 Kb3 Ke7 Ka4 Bb6 Kb4 Ke8 Nc6 Kd7 Kc3 Ke8 Kd3 Bc5 Ke2 Kf8 Kf3 Ba3 Bb3 Bc5 Ne5 Ke7 Nd3 Bb6 Nb4 Bc5 Nd5 Kf8 Bc2 Ba7 h4 Kg7 h5 Bc5 Ke2 Nc4 Bf5 Nb6 Nc3 Bb4 Nb5 Kf6 Bc2 Nc4 Nd4 Bc5 Nf5 Nd6 Ng3 Ke5 Nf1 Bb4 Nh2 Kf6 Bb3 Bc5 Nf3 Bb6 Kd3 Bc5 Bc2 Ke6 Ng1 Ke5 Ne2 Bb6 Bb3 Bc5 Nc3 Kf6 Nd5 Ke6 f4 gxf4 g5 fxe3 gxh6 e2 Nf4 Ke7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Bb7 Ne5 O-O O-O Na6 e4 c5 exd5 exd5 Re1 cxd4 Bxd4 Nc5 Ng4 dxc4 Nxc4 Bxg2 Kxg2 Nxg4 Qxg4 Bf6 Rad1 Bxd4 Rxd4 Qc7 Nd6 Ne6 Rxe6 h5 Qe4 fxe6 Qxe6 Kh7 Rd5 g6 Ne4 Rad8 Ng5 Kg7 Qe4 Rfe8 Qd4 Kg8 Rxd8 Rxd8 Qf6 Rd6 Qe5 Qc6 Kh3 Qd7 Kg2 Qc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 O-O Nbd7 Rc1 c5 Bg3 a6 cxd5 Nxd5 Nxd5 Bxd5 dxc5 Nxc5 b4 Ne4 Bc7 Qe8 a3 a5 Bd3 axb4 axb4 Bxb4 Bxe4 Bxe4 Qd4 Bxf3 Qxb4 Be2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Nbd7 O-O O-O Re1 c5 e4 dxe4 Nxe4 Bb7 Nfg5 cxd4 Bxd4 Qc7 Nxf6 Bxf6 Bxb7 Qxb7 Ne4 Bxd4 Qxd4 Rad8 Rad1 Qa8 Qc3 Nb8 Nf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 Qd2 Nc6 e3 O-O Rc1 a6 Be2 dxc4 Bxc4 e5 d5 Na7 Qc2 Nb5 Nxb5 axb5 Bb3 e4 Nd4 Bxd4 exd4 c6 dxc6 Qxd4 O-O bxc6 Qxc6 Bd7 Qd5 Qxd5 Bxd5 Ra6 Rfd1 Be6 a3 Bxd5 Rxd5 Rb8 Rd4 Ra4 Rcd1 Rc8 Kf1 Rc2 R4d2 Rxd2 Rxd2 Rc4 Ke2 b4 Kd1 bxa3 bxa3 Ra4 Ra2 f5 Kc2 f4 Kb3 Rd4 Ra1 Kf7 a4 e3 Kc3 Rd8 fxe3 fxe3 Re1 Ra8 Kb3 Rb8 Kc2 Ra8 Rf1 Ke6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 Nc3 Bg7 d4 cxd4 Nxd4 Bxg2 Kxg2 O-O e4 Qc7 b3 Nxe4 Nxe4 Qe5 Qf3 Qxd4 Rb1 Qe5 Bf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 Qd2 dxc4 e4 c5 d5 exd5 e5 Bg5 Qxd5 Nc6 Bxc4 O-O O-O Qxd5 Bxd5 Nb4 Nxg5 Nxd5 Nxd5 hxg5 f4 gxf4 Rxf4 Rd8 Nc7 Rb8 Raf1 Rd7 Nb5 Re7 Nxa7 Bd7 a4 Ra8 Nb5 Bxb5 axb5 Ra5 b6 Rb5 b4 cxb4 Rb1 b3 Rf3 b2 Rf2 Rexe5 Rfxb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 a4 Bd7 Qxc4 Bc6 Bg5 a5 Nc3 Na6 Rac1 Qd6 Ne5 Bxg2 Kxg2 c6 Bxf6 gxf6 Nf3 Rfd8 Rfd1 Qb4 Qa2 Rd7 e3 Rad8 Rc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O Rc1 dxc4 e3 c5 Bxc4 cxd4 Nxd4 Bd7 O-O Nc6 Nb3 Rc8 Be2 Nd5 Bxe7 Ncxe7 Nxd5 Nxd5 Rxc8 Qxc8 Qd4 Qb8 Bf3 Nf6 Nc5 Bb5 Rd1 b6 Ne4 Nxe4 Bxe4 Rc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 Nc6 d4 cxd4 Nxd4 e6 g3 Qb6 Nb3 d5 cxd5 Nxd5 Bg2 Nxc3 bxc3 Be7 O-O O-O Be3 Qc7 Nd4 Rd8 Qa4 Bd7 Nxc6 Bxc6 Bxc6 bxc6 c4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Rc1 Bb7 Be2 Nbd7 cxd5 exd5 O-O c5 dxc5 bxc5 Rc2 Rc8 Rd2 Qb6 Qb3 Rfd8 Rfd1 Qxb3 axb3 Nb6 Ne5 Kf8 h3 a6 Bf3 Na8 Ng4 Ng8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 Nc6 d4 cxd4 Nxd4 e6 g3 Qb6 Nb3 d5 cxd5 Nxd5 Bg2 Nxc3 bxc3 Be7 O-O e5 Be3 Qc7 Nc5 O-O Qa4 Bxc5 Bxc5 Rd8 Rfd1 Be6 h3 Rxd1 Rxd1 Rd8 Rxd8 Qxd8 Bxa7 Qa8 Bxc6 bxc6 Kh2 h5 Qa5 f6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Qc2 c5 dxc5 dxc4 Bxc4 Qa5 O-O Bxc3 Qxc3 Qxc3 bxc3 Nd7 c6 bxc6 Rab1 Nb6 Be2 c5 Rfc1 Bb7 Kf1 Bd5 Rb5 Nd7 Ra5 Rfb8 c4 Bc6 Ne1 Rb4 Bd1 Rb7 f3 Rd8 Nd3 g5 Bb3 Kf8 Nxc5 Nxc5 Rxc5 Rd6 Ke2 Ke7 Rd1 Rxd1 Kxd1 Kd6 Ra5 f5 Ke2 h5 e4 fxe4 fxe4 Bxe4 Rxg5 Bf5 Ke3 h4 Kd4 e5 Kc3 Bb1 a3 Re7 Rg4 h3 g3 Re8 Rg7 Rf8 Rxa7 Rf2 Kb4 Rxh2 c5 Kc6 Ba4 Kd5 Rd7 Ke4 c6 Rb2 Ka5 Rb8 c7 Rc8 Kb6 Ke3 Bc6 h2 g4 Rh8 Rd1 Ba2 Re1 Kf4 Re4 Kg3 Rxe5 Kxg4 Re2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Bg4 Re1 Be7 c4 Nf6 cxd5 Bxf3 Qxf3 Qxd5 Qh3 Nxd4 Nc3 Qd7 Qxd7 Kxd7 Be3 Ne6 Rad1 Bd6 Bf5 Ke7 Nb5 Rhd8 Nxd6 cxd6 h3 b6 g4 h6 Bd4 Rac8 Bc3 g6 Bc2 h5 f3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Be2 Bb7 a3 b4 Na4 bxa3 bxa3 Be7 O-O O-O Bb2 c5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Be7 O-O Nc6 Re1 Bg4 c4 Nf6 Nc3 dxc4 Bxc4 O-O Be3 Bxf3 Qxf3 Nxd4 Bxd4 Qxd4 Rxe7 Qxc4 Qxb7 c6 Qb3 Qxb3 axb3 Rab8 Ra3 Rfe8 Rxe8 Rxe8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Rc1 Bb7 Be2 Nbd7 cxd5 exd5 O-O c5 Qa4 a6 dxc5 bxc5 Rfd1 Qb6 Qb3 Qa7 Bg3 Rad8 Ne1 d4 exd4 cxd4 Na4 Rc8 Rxc8 Rxc8 Bc4 Rf8 Qd3 Bc6 Bb3 Ne4 Qxd4 Qb7 Bd5 Ndf6 Bxc6 Qxc6 b3 Re8 Qd3 h5 Qc4 Qb7 Nf3 Rc8 Qe2 Ng4 Ne5 Nxg3 hxg3 Qb5 Nc4 Bf6 Nab6 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 Qc2 Nd7 Nxd5 exd5 Bg5 f6 Bf4 c5 g3 g6 h4 Qe7 Bg2 Bg7 h5 f5 Qd2 Bf6 Rc1 Rc8 Rc3 Rc6 Re3 Re6 Rxe6 Qxe6 Ng5 Qe7 dxc5 Nxc5 hxg6 d4 g7 Bxg7 Bxb7 Qxb7 f3 Qd5 Rxh7 Rxh7 Nxh7 Qb3 Bd6 Ne6 Ng5 Bh6 Bf4 Bxg5 Bxg5 Nxg5 Qxg5 Qxb2 Qxf5 Qc1 Kf2 Qe3 Kf1 Qc1 Kg2 Qxa3 Qh5 Kd7 Qg4 Kc6 Qxd4 b5 g4 b4 g5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 e3 Nbd7 Qc2 Bd6 e4 e5 cxd5 cxd5 exd5 exd4 Nxd4 O-O Be2 Nb6 O-O Nbxd5 Nxd5 Nxd5 Rd1 Qe7 Bf3 Re8 g3 Bh3 Bd2 Bc5 Bxd5 Bxd4 Bc3 Bxc3 Qxc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Rc1 dxc4 Bxc4 Nbd7 O-O a6 a4 c5 Qe2 cxd4 exd4 Nh5 Bxe7 Qxe7 d5 Nf4 Qe3 Qf6 Ne4 Qf5 Ng3 Qf6 Ne4 Qf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bg5 e6 Qd2 Be7 O-O-O O-O f4 h6 Bh4 e5 Nf5 Bxf5 exf5 exf4 Kb1 d5 Bxf6 Bxf6 Nxd5 Be5 g3 fxg3 hxg3 Ne7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Rc1 dxc4 Bxc4 Nbd7 O-O c5 dxc5 Nxc5 Qe2 a6 Rfd1 Qe8 Ne5 b5 Nxb5 axb5 Bxb5 Ba6 Rxc5 Bxc5 Bxa6 Qa4 Bxf6 gxf6 Bb5 Qxa2 Nd7 Be7 Qg4 Kh8 Nxf8 Bxf8 Qf3 Be7 Bc4 Qa7 Qh5 Kg7 Qg4 Kf8 Bf1 Rd8 Rc1 Qb8 Rc2 f5 Qe2 Kg7 g3 Rc8 h3 Rxc2 Qxc2 Bf6 b3 Qb4 Qd1 Qc3 Kg2 Qc6 Kh2 Qc5 Be2 Be7 Kg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O Nb3 a5 a4 d5 exd5 Nxd5 Bxe7 Ncxe7 Nb5 Bd7 Be2 Nf5 N3d4 Nxd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qc2 Rfd8 Rfc1 Rab8 a4 Qb6 dxc5 Nxc5 Bxc6 Qxc6 Nb5 Be7 Qf5 Qe8 Ne5 Rb7 Nd4 Rc7 Nb5 Rb7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Bg5 Be7 Nc3 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 O-O Nd7 b4 c5 bxc5 bxc5 Rb1 Bc6 Bb5 Qc7 Qc2 Rfd8 Rfc1 Rab8 a4 Qd6 dxc5 Nxc5 Bxc6 Qxc6 Nb5 Be7 Nxa7 Qa6 Nb5 Qxa4 Qxa4 Nxa4 Nfd4 Rd7 Nc6 Rb6 Nbd4 Bf6 Rxb6 Nxb6 Rb1 Na4 g3 Nc5 Kg2 g6 Nf3 Rd6 Ncd4 Ne6 Rb8 Kg7 Nb3 Rd7 Rb5 Bc3 Nc1 Ng5 Ne2 Ne4 Rb3 Rc7 Rb5 Rd7 Nf4 Nf6 h3 h5 Rb3 Ne4 Ra3 Kg8 Ra4 Nf6 g4 hxg4 hxg4 Nxg4 Ne2 d4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Qa5 Qd2 cxd4 Nxd4 Bxd4 exd4 Bc6 Nb5 Qd8 O-O a6 Na3 Re8 Nc2 Rxe2 Qxe2 Bb5 Rxb5 axb5 Qxb5 Rxa2 Ne3 Ra5 Qb7 Qe8 Nxd5 Rb5 Qa8 Qd7 Nc3 Rb4 d5 Qc7 Nd1 Rb5 Ne3 Qa5 Qxa5 Rxa5 Rd1 Nd7 g4 g6 Kg2 Ra4 h3 Kg7 d6 Ra6 f4 Rc6 h4 Kf8 g5 hxg5 hxg5 f5 Rd4 Kf7 Nc4 Ke6 Kf3 Rc5 Ke3 Rb5 Kd2 Rd5 Rxd5 Kxd5 Ne5 Kxd6 Nxg6 Nc5 Nh4 Ke6 Ke3 Ne4 Nf3 Kf7 Kd4 Ke6 Kc4 Nf2 Kd4 Ne4 Ne1 Kd6 Nc2 Nc5 Ke3 Ne6 Nd4 Ng7 Kd2 Kc5 Kd3 Kd5 Ne2 Nh5 Ke3 Ng7 Ng3 Kd6 Kf3 Ke7 Ne2 Ne6 Ng3 Ng7 Nf1 Kf7 Ne3 Kg6 Nd5 Ne6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Be7 O-O Nc6 c4 Nb4 Be2 dxc4 Bxc4 O-O Nc3 Nd6 Bb3 Bf6 h3 Bf5 Be3 Re8 a3 Nd3 Rb1 c5 dxc5 Ne4 Bc2 Nxb2 Qxd8 Raxd8 Rxb2 Bxc3 Rxb7 Nxc5 Bxc5 Bxc2 Rxa7 Bd1 Re7 Rxe7 Bxe7 Rd3 Ng5 Bb2 Bb4 h6 Ne4 f5 Nc5 Rd5 Re1 f4 a4 Rd4 a5 Rxb4 Rxd1 Bd4 Ne6 Ba7 Rd7 Rb1 Kh2 Bxf2 Nxf4 Ra1 Ne6 Rxa5 Rxg7 Kh8 Rf7 Be3 Kg3 Bd2 Rd7 Bc3 Kf3 Kg8 Nf4 Rf5 Ke4 Rf7 Rd8 Kh7 Rd3 Re7 Kf3 Bb2 Rb3 Bc1 Nd5 Re5 Nf6 Kg6 Ne4 Rf5 Ke2 Re5 Rb4 Re7 Rc4 Re8 g3 Bb2 Kf3 Re6 Rc5 Bd4 Rd5 Be5 Rb5 Bc7 Rc5 Bb6 Rc8 Bd4 Rg8 Bg7 h4 Ra6 Kf4 Ra5 Re8 Rf5 Ke3 Re5 Rg8 Re7 Kf4 Rf7 Kg4 h5 Kh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qc2 Rfc8 Rfc1 Bxb5 Nxb5 Qc6 dxc5 Nxc5 Qf5 Qe6 Nfd4 Qxf5 Nxf5 Ne6 Rxc8 Rxc8 Nxa7 Rc2 Nb5 Rxa2 h3 Ra5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 Qe1 b5 Bf3 Bb7 e5 Ne8 f5 dxe5 fxe6 Bxf3 exf7 Rxf7 Nxf3 Nd7 Bg5 Bf8 a3 Nd6 Nd2 Rxf1 Qxf1 Qc6 Re1 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 a4 h6 Nbd2 exd4 cxd4 Nb4 Qe2 Bf8 e5 Bc6 axb5 Bxb5 Qd1 Nfd5 Ne4 c6 Nc3 Rb8 Nxb5 axb5 exd6 Bxd6 Bd2 Qc7 Qb1 Qd7 Ne5 Bxe5 Rxe5 Rxe5 dxe5 c5 Qe4 c4 Bd1 Nd3 Bg4 Qb7 Qd4 Qb6 Qxd5 Qxf2 Kh2 Qxd2 Rf1 Qg5 Qxf7 Kh8 e6 Ne5 Qf5 Nxg4 hxg4 Re8 Qxg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bf3 Rb8 Qd2 Nxd4 Bxd4 e5 Ba7 Ra8 Be3 Bd7 a5 Rac8 Be2 Bc6 Qd3 Qd8 Rfd1 exf4 Bxf4 Bf8 Bf3 Qe7 Bxd6 Qxd6 Qxd6 Bxd6 Rxd6 Nxe4 Rxc6 Rxc6 Nxe4 Rce6 Kg1 Rxe4 Bxe4 Rxe4 Rd1 g5 Rd5 h6 c3 Re6 Kf2 Kg7 g4 b6 h3 Kg6 Kf3 h5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 Qd7 axb5 axb5 Rxa8 Bxa8 d5 Nd8 Nf1 h6 N3h2 Nb7 Bc2 Nc5 b4 Na6 Ng4 Nh7 Ng3 c6 dxc6 Bxc6 Bb3 Nc7 Qf3 Ne6 h4 Qd8 Rd1 Qa8 Bd5 Bxd5 exd5 Nc7 Ne4 Qc8 Ne3 Qd7 Nf5 Ra8 Qh3 Rd8 Be3 Qc8 Qf3 Ne8 Bb6 Rd7 h5 Qb7 Be3 Kh8 g4 Be7 Nxe7 Rxe7 g5 hxg5 Bxg5 Rc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 d4 d5 Nc3 c6 Bg5 Nbd7 e3 Qa5 cxd5 Nxd5 Qd2 N7b6 Nxd5 Qxd2 Nxd2 exd5 Bd3 a5 a4 Bb4 Ke2 Bg4 f3 Bh5 h4 O-O g4 Bg6 b3 Bxd3 Kxd3 Rfe8 Rac1 c5 Bf4 Rac8 dxc5 Nd7 c6 bxc6 Rhd1 Nc5 Kc2 f6 Nf1 Ne6 Bg3 Red8 Bf2 c5 Nd2 c4 bxc4 Nc5 e4 d4 Nb1 d3 Kb2 d2
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nf6 Nc3 O-O h3 dxc4 Bxc4 Na5 Bd3 Be6 Re1 Nc6 a3 a6 Bf4 Qd7 Ne5 Nxe5 dxe5 Nd5 Nxd5 Bxd5 Qc2 g6 Rad1 c6 Bh6 Rfd8 e6 fxe6 Bxg6 Bf8 Bxf8 Rxf8 Be4 Rf7 Re3 Rg7 Rdd3 Rf8 Rg3 Kh8 Qc3 Rf7 Rde3 Kg8 Qe5 Qc7 Rxg7 Rxg7 Bxd5 Qxe5 Bxe6 Qxe6 Rxe6 Rd7 b4 Kf7 Re3 Rd1 Kh2 Rc1 g4 b5 f4 c5 bxc5 Rxc5 Rd3 Ke7 Kg3 a5 Kf3 b4 axb4 axb4 Ke4 Rb5 Rb3 Rb8 Kd5 Kf6 Kc5 Re8 Rxb4 Re3 h4 Rh3 h5 Rh4 f5 Rh1 Kd5 Rd1 Rd4 Re1 Kd6 Re8 Kd7 Rg8 h6 Kf7 Rc4 Kf6 Re4 Kf7 Kd6 Kf6 Re6 Kf7 Re7 Kf6 Rg7 Rd8 Kc5 Rd5 Kc4 Rd4 Kc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 g6 Bg5 Bg7 e3 O-O Bd3 Be6 Qe2 Nbd7 O-O h6 Bh4 Bg4 cxd5 cxd5 h3 Bxf3 Qxf3 e6 Rfc1 a6 Rc2 Rc8 Rac1 Nb6 b3 Qe7 Qe2 Qa3 Qe1 Qd6 Bg3 Qe7 Qd1 Rfd8 Na4 Rxc2 Qxc2 Nxa4 bxa4 Ne8 Rb1 h5 a5 h4 Bf4 Bf6 Qb3 Rd7 Rc1 Nd6 Qb6 Kg7 Qb4 Ne4 a3 Qxb4 axb4 Nd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 c5 c4 e6 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 a3 Be6 Kh1 Qd7 Nxe6 fxe6 f4 Red8 Bg1 Rac8 Qa4 Kh8 Rad1 Qe8 e4 d4 Ne2 Bc5 Qb5 Bb6 h3 e5 fxe5 Nxe5 Qxe8 Rxe8 Nxd4 Nc4 e5 Rxe5 Bxb7 Rc7 Rc1 Nxb2 Rxc7 Bxc7 Nc6 Re2 Nd4 Re5 Nf5 Bb6 Nxh6 Ra5 Bxb6 axb6 Nf5 Rxa3 Kh2 Nc4 g4 Ra7 Bh1 Ne5 g5 Nh5 Re1 Ra5 Nd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 Nbd7 e3 Qa5 cxd5 Nxd5 Qd2 Bb4 Rc1 O-O Bd3 e5 O-O exd4 exd4 f6 Bh4 Rd8 a3 Bxc3 bxc3 Nf8 Bg3 Be6 Rfe1 Bf7 c4 Qxd2 Nxd2 Nb6 Nb3 Na4 Bf1 Rd7 Na5 Ne6 d5 Nd4 dxc6 Nxc6 Nxc6 bxc6 c5 Re8 Rxe8 Bxe8 Bd6 Bf7 Rb1 Bd5 Rb8 Kf7 Rf8 Ke6 g3 g6 Ba6 Rxd6 cxd6 Kxd6 Rxf6 Ke5 Rf8 c5 Re8 Kd4 Rd8 Ke5 f4 Ke4 Bf1 Bb3 Kf2 Nb2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 e6 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Be2 dxc4 Bxc4 a6 Qe2 b5 Bd3 Bb7 O-O Nbd7 e4 Nh5 Bd2 Qc7 g3 Rad8 Be3 Bxe3 Qxe3 Qc5 Rfe1 Nhf6 a3 Ng4 Qxc5 Nxc5 Bc2 f5 Ng5 f4 Rad1 Rxd1 Bxd1 Ne5 gxf4 Ned3 b4 h6 bxc5 hxg5 Re3 Nxf4 a4 b4 Ne2 Rc8 Bb3 Rxc5 Nxf4 gxf4 Bxe6 Kf8 Re1 Re5 Bb3 Rxe4 Rd1 Ke7 Rc1 Kf6 Kf1 a5 h3 g5 Rc7 Re7 Rc5 Re5
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Be7 exf6 Bxf6 Bxf6 Qxf6 g3 Na6 Bg2 Bb7 Ne4 Qe7 O-O O-O-O a4 Kb8 Qd2 b4 Rac1 f5 Ng5 e5 Rxc4 c5 Bxb7 Qxb7 Qe3 exd4 Qe5 Ka8 Ne6 Qh7 h4 Rhe8 Qe2 Rd6 Qf3 Qb7 Qxb7 Kxb7 Nxc5 Nxc5 Rxc5 d3 Rd1 Re2 Rb5 Ka6 Rxb4 d2 Kf1 Rde6 Kg2 Rd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O Rc1 Ne4 Bxe7 Qxe7 e3 c6 Bd3 Nxc3 Rxc3 dxc4 Bxc4 Nd7 O-O b6 Bd3 c5 Bb5 Rd8 Bc6 Rb8 Qc2 cxd4 Nxd4 e5 Nf5 Qf6 Rd1 Nc5 Rxd8 Qxd8 Ng3 Be6 b4 Rc8 Bf3 Na6 a3 Rxc3 Qxc3 Qc7 Qd2 Nb8 Ne4 Nd7 h3 Nf6 Nxf6 gxf6 e4 Kg7 Be2 Qc6 Qe3 Qc2 Kh2 Qb3 Qxb3 Bxb3 Bg4 Bc2 Bf5 Bd3 Kg3 Be2 Bg4 Bf1 Kf3 Kf8 a4 Bc4 Ke3 Ke7 h4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 Nbd7 e3 Qa5 Nd2 Bb4 Qc2 O-O a3 dxc4 Bxf6 Nxf6 Nxc4 Bxc3 Qxc3 Qxc3 bxc3 c5 Be2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 c5 c4 e6 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 a3 Be6 Kh1 Bg4 f3 Bh5 Bg1 Qd7 Qa4 Bc5 Rad1 Bb6 Rfe1 Bg6 Qb5 Rad8 e3 Qd6 Nce2 Ne5 Qb3 Ba5 Nc3 Nd3 Re2 Nc5 Qa2 Bxc3 bxc3 Qa6 Red2 Na4 Qb3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 Nbd7 e3 Qa5 cxd5 Nxd5 Qd2 Bb4 Rc1 e5 a3 Bd6 dxe5 Nxe5 Nxe5 Bxe5 b4 Bxc3 Qxc3 Nxc3 bxa5 Ne4 Bf4 O-O f3 Nf6 e4 Re8 Kf2 a6 Be2 Be6 Rb1 Re7 Rhd1 Rae8 Rb2 Bc8 Rbd2 Rd7 Rxd7 Nxd7 g4 Nc5 Be3 Nd7 g5 Ne5 Bd4 Ng6 Kg3 Nf8 h4 Rd8 f4 Be6 Bc3 Rxd1 Bxd1 Nd7 f5 Bc4 h5 h6 gxh6 gxh6 e5 Nc5 Kf4 Bd5 Bc2 f6 e6 Kg7 Bb4 Nb3 Ke3 c5 Bc3 Kf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 Nf3 c5 c4 e6 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 a3 Be6 Kh1 Bg4 f3 Bh5 Nxc6 bxc6 Na4 Qc8 Bd4 Qe6 Rc1 Nd7 Rc3 Bf6 e3 Bg6 Kg1 Be7 Qd2 Rab8 Re1 a5 Bf1 h5 Rec1 Ne5 Bxe5 Qxe5 Rxc6 Bf6 R6c5 Qxe3 Qxe3 Rxe3 Rxd5 Rxf3 Be2 Re3 Bxh5 Bxh5 Rxh5 g5 Nc3 Rd8 Rc2 Kg7 Kg2 Kg6 g4 Rd4 h3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nc6 c4 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5 Nc3 Bb4 Bd2 Bxc3 bxc3 Qd6 Rb1 b6 f4 exf4 e4 Nge7 Qf3 O-O Bxf4 Qa3 Be2 f5 O-O fxe4 Qxe4 Qxc3 Be3 Qa3 Bd3 Qd6 Qxh7 Kf7 Rb5 Nxd4 Qe4 Rad8 Bxd4 Qxd4 Rf5 Nxf5 Qxf5 Kg8 Qh7 Kf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 c5 c4 e6 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 a3 Be6 Nxe6 fxe6 Qa4 Rc8 Rad1 Kh8 Kh1 a6 f4 Na5 f5 b5 Qh4 Ng8 Qh3 Nc4 Bc1 Bg5 fxe6 Bxc1 Rxc1 Ne3 Nxd5 Nxf1 Rxf1 Rf8 Nf4 Ne7 Qg4 g5 Qh3 Rf6 Nd3 Rxf1 Bxf1 Kg7 Qg4 Qd5 e4 Qd4 h4 Rf8 Be2 Qe3 Kg2 Ng6 h5 Ne7 b4 Kh7 Kh2 Rd8 e5 Rxd3 Bxd3 Qxd3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 Nbd7 e3 Qa5 Nd2 Bb4 Qc2 O-O Be2 e5 Bxf6 Nxf6 dxe5 Ne4 cxd5 Nxc3 bxc3 Bxc3 Rc1 Bxe5 dxc6 bxc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb7 Bg2 Bb4 Bd2 a5 O-O O-O Bg5 Be7 Nc3 Ne4 Bxe7 Qxe7 Qc2 Nxc3 Qxc3 d6 Ne1 Bxg2 Nxg2 c5 Qf3 Ra7 Rad1 a4 Rd2 axb3 axb3 Qc7 d5 e5 e4 Nd7 Qd3 Rfa8 Ne3 Ra1 Nd1 Nf6 Nc3 Ne8 Kg2 Rxf1 Qxf1 Qb8 Qb1 Nc7 Ra2 Kf8 Qa1 Rxa2 Qxa2 Qa8 Qxa8 Nxa8 f4 f6 Na4 g6 Kf3 Kf7 h4 h6 g4 g5 fxg5 fxg5 h5 Ke7 Nc3 Nc7 Nd1 Ne8 Ne3 Ng7 Ke2 Kd7 Kd3 Kc7 Kc2 Kc8 b4 Kc7 Kb3 Kb7 Ka4 Kb8 bxc5 bxc5 Ka5 Kb7 Kb5 Kc7 Ka6 Kc8 Kb6 Kd7 Kb7 Ne8 Nf5 Nf6 Nxh6 Nxe4 Nf5 Nf6 h6 e4 Kb6 Nh7 Kb5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 c5 O-O Nc6 Ne5 Bd7 Na3 cxd4 Naxc4 Rc8 Qb3 Nxe5 Nxe5 Bc6 Nxc6 bxc6 Rd1 c5 e3 Bd6 exd4 c4 Qb5 Qd7 a4 O-O Be3 Rc7 d5 e5 Rdc1 Rfc8 Bf1 g6 Bxc4 Qxb5 Bxb5 Nxd5 Ba6 Rxc1 Rxc1 Rxc1 Bxc1 Bc5 Bc4 Nb6 Bb3 Kf8 a5 Nd7 Ba4 Nf6 Bd2 Nd5 Bb3 Nb4 Bc4 Ke7 Kg2 h5 f3 Nc6 Bd5 Nb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 cxd5 exd5 Bg5 Be7 Qc2 g6 e3 Bf5 Bd3 Bxd3 Qxd3 O-O Bxf6 Bxf6 b4 Qd6 Rb1 Nd7 O-O Rfd8 Rfc1 Nb6 Nd2 Qe7 Qc2 a6 a4 Bg7 Ne2 Bh6 Re1 Qe6 Nc1 Nc4 Qc3 Nd6 Nd3 Bg7 Nc5 Qe7 Re2 Re8 Rbe1 Qc7 a5 Re7 Qd3 Rae8 f3 Nf5 g3 h5 Kg2 Qc8 Ndb3 Rd8 Rd1 Bf6 Qc3 Bg7 e4 dxe4 fxe4 h4 g4 Nxd4 Nxd4 Qxg4 Kf1 Rd6 Rd3 Qf4 Rf2 Qe5 Rfd2 Qf4 Rf2 Qe5 Rfd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 c5 O-O Nc6 Ne5 Bd7 Na3 cxd4 Naxc4 Nd5 Nxc6 Bxc6 Qxd4 Nb4 Bxc6 Nxc6 Qc3 f6 Be3 Be7 Rfd1 Qc7 Qb3 g5 Rac1 Rd8 Rxd8 Bxd8 Qxb7 Qxb7 Nd6 Kd7 Nxb7 Bb6 Nc5 Bxc5 Bxc5 f5 Rd1 Kc7 Rd6 Re8 b4 a6 f3 h5 h4 gxh4 gxh4 Rg8 Kf2 Rg6 a4 f4 b5 axb5 axb5 Ne7 Ra6 Nf5 Ra7 Kc8 b6 Rg7 Ra4 Kb7 Rxf4 Kc6 Rc4 Kb5 Rb4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 b3 Bg7 Bb2 O-O Nc3 Na6 d4 d5 cxd5 Nxd5 Nxd5 Qxd5 Nh4 Qd7 dxc5 Qxd1 Rfxd1 Bxb2 Bxb7 Nxc5 Bxa8 Bxa1 Rxa1 Rxa8 Rd1 a5 Nf3 a4 bxa4 Rxa4 Rd2 Ne4 Rc2 b5 Nd2 Nd6 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O Qc2 dxc4 Qxc4 a6 Bf4 Bd6 Ne5 Nd5 Nc3 Nxf4 gxf4 Nd7 e3 Qe7 O-O Rb8 Ne4 Nf6 Nc5 c6 b4 Kh8 a4 Nd5 b5 f6 Ned3 axb5 axb5 cxb5 Qxb5 Rd8 Ra7 Bxc5 Nxc5 Bd7 Nxd7 Rxd7 f5 g6 fxe6 Qxe6 Rfa1 Kg7 Qb3 Nf4 Qxe6 Nxe6 Rb1 Nd8 Bf3 Rc8 Ra5 Rcc7 Rab5 f5 h4 Kf7 h5 Kg7 Kg2 Re7 Rb6 Rf7 Bd5 Rfd7 R1b5 Re7 Kg3 Red7 hxg6 hxg6 Kf4 Rc2 Kg5 Rxf2 Rxg6 Kf8 Bb3 Nf7 Kf6 f4 e4 Rb2 e5 f3 e6 f2 Bc4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 g3 dxc4 Qa4 Nbd7 Bg2 a6 Nc3 Rb8 Qxc4 b5 Qd3 Bb7 Bf4 Rc8 O-O c5 dxc5 Bxc5 Rad1 O-O Ne5 Bxg2 Kxg2 Nxe5 Bxe5 Be7 Qf3 Qa5 Qb7 Rfe8 a3 b4 Bxf6 gxf6 axb4 Qxb4 Qxb4 Bxb4 Ne4 Kg7 Nd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 cxb5 a6 e3 g6 Nc3 Bg7 a4 O-O e4 axb5 Bxb5 Ba6 Bd2 e6 Nge2 exd5 exd5 d6 O-O Nbd7 Bf4 Ne5 Bxe5 dxe5 Qd3 Qa5 Ng3 Rfd8 Nge4 Nxe4 Nxe4 Bxb5 Qxb5 Rxd5 Rfc1 h5 h4 Rd4 Qxa5 Rxa5 Rxc5 Raxa4 Rc8 Bf8 Rxa4 Rxa4 Nc5 Ra1 Kh2 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bf4 O-O e3 c5 dxc5 Ne4 Qb3 Na6 cxd5 Naxc5 Qc4 b5 Nxb5 Bxb2 Bc7 a6 Bxd8 axb5 Qc2 Bc3 Qxc3 Nxc3 Bxe7 Nb3 Rd1 Rxa2 Bxf8 Kxf8 Nd4 Nxd1 Nxb3 Nxf2 Rg1 Ng4 d6 Nxe3 Nc5 Bg4 h3 Rc2 Na6 Be6 Be2 Bc4 Bf3 Nf5 d7 Ke7 Nb8 Rc1 Kf2 Rxg1 Kxg1 Nd4 Be4 f5 Bb1 Be6 Kf2 Bxd7 Ke3 Nc6 Nxd7 Kxd7 g4 fxg4 hxg4 Ke6 Kf4 Kd5 Ba2 Kd4 Kg5 Ne5 Bg8 b4 Bxh7 b3 Bg8 b2 Ba2 Kc3 Bb1 Kd2 Kf4 Kc1 Be4 Nc4 Bxg6 Na3 g5 Nc2
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Na6 e4 Bg4 Bxc4 e6 Be3 Nb4 Be2 Be7 O-O O-O Ne5 Bxe2 Qxe2 Qa5 Nc4 Qa6 b3 c5 dxc5 Qc6 Bg5 Bxc5 Rad1 Nd7 Qd2 f6 Be3 Rfd8 Bxc5 Qxc5 Nb5 Nc6 Ncd6 Rab8 Rc1 Qb6 b4 Nf8 Qb2 Rd7 Nc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Nxc6 bxc6 e5 Nd5 Ne4 Qc7 f4 Qb6 c4 Bb4 Ke2 f5 Nf2 Ba6 Kf3 Ne7 Be3 Bc5 Bxc5 Qxc5 Qd6 Qb6 b3 c5 Rd1 Bb7 Ke3 Kf7 Qxb6 axb6 Rxd7 Bc6 Rd2 g5 Nh3 h6 fxg5 Ng6 gxh6 Nxe5 Be2 Rxh6 Nf4 Rg8 h3 Ke7 Rhd1 Ng6 Bf3 Bxf3 Kxf3 Nxf4 Kxf4 Rhg6 Ke5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 Nf6 Nc3 c6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 g3 Bb7 Bg2 Qb6 exf6 c5 d5 O-O-O O-O b4 Na4 Qb5 a3 Nb8 Qg4 Bxd5 Bxd5 Rxd5 axb4 cxb4 Be3 Nc6 Rfc1 Ne5 Qe4 Qc6 Kg2 Bd6 Bxa7 Kb7 Nb6 Rc5 Qxc6 Rxc6 f4 c3 bxc3 Nd3 Rd1 Nb2 Rdb1 Nd3 Na4 e5 fxe5 Bxe5 Rd1 bxc3 Rxd3 c2 Rb3 Kxa7 Nc3 Ra6 Rc1 Rc8 Rxc2 Ka8 h4 Bxf6 Kh3 Bg7 Ra2 Rxa2 Nxa2 Be5 h5 Rh8 Rb5 f6 Kh4 Rd8 h6 Rg8 g4 Rh8 Rxe5 fxe5 g5 e4 Kg4 e3 Nc3 Kb7 Kh5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nf6 Nc3 d5 cxd5 Nxd5 e4 Nb4 Bc4 Nd3 Ke2 Nf4 Kf1 Ne6 b4 cxb4 Ne2 Nc7 d4 e6 h4 Nd7 Bg5 Be7 Rc1 Nb6 Bd3 f6 Bf4 Na6 h5 O-O Qb3 Bd7 Ng3 Kh8 Nh4 Rf7 Be3 Rc8 Rxc8 Nxc8 Ng6 Kg8 Nf4 Nc7 e5 Nb6 exf6 gxf6 Bg6 hxg6 hxg6 Rg7 Qd1 Be8 Qh5 Bxg6 Nxg6 Qe8 Nxe7 Qxe7 Qh8 Kf7 Nh5 Qf8 Nxg7 Qxh8 Rxh8 Kxg7 Rb8 Nbd5 Rxb7 a5 Bf4 Nxf4 Rxc7 Kg6 Ra7 Kf5 Rxa5 Ke4 Ra6 Kxd4 g3 Nd5 Rxe6 Nc3 Ra6 Kd3 g4 Kc2 f4 Kb2 g5 fxg5 fxg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 Nc3 cxd4 Nxd4 Nc6 g3 Bc5 Nb3 Bb4 Bg2 d5 cxd5 Nxd5 O-O Nxc3 Qxd8 Kxd8 bxc3 Bxc3 Rb1 a5 Be3 Bb4 Nc5 Ke7 a3 Bxa3 Nxb7 Bd7 Nc5 Rhb8 Nxd7 Rxb1 Rxb1 Kxd7 Rb7 Kd6 Rxf7 a4 Rf4 Ra6 Rc4 Ne5 Rd4 Ke7 Bb7 Ra5 Bg5 Kf7 Bd2 Ra7 Bc8 Be7 Re4 Ra8 Bb7 Ra7 Bc8 Ra8 Bb7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Nxd7 O-O Ngf6 Qe2 e6 c3 Rc8 d4 cxd4 cxd4 d5 e5 Ne4 Be3 Nb6 Ne1 Nc4 Nd3 f5 f3 Ng5 Nc3 Nf7 Nf4 Rc6 g4 g6 gxf5 Qg5 Kh1 Qxf5 Nfxd5 exd5 Nxd5 Bg7 Rac1 Ncxe5 Bf4 Rxc1 Rxc1 Qd3 Rc8 Kd7 Rc7 Ke6 Bxe5 Qb1 Kg2 Bxe5 Nc3
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 d6 d4 exd4 Qxd4 Nf6 g3 Nc6 Qd2 Be6 Nd5 Ne5 b3 Ne4 Qe3 Nc5 Bb2 c6 Nf4 Ng4 Qd4 Ne4 Bh3 Qa5 Kf1 Ngxf2 Bxe6 fxe6 Nxe6 Kd7 Nh3 Nxh3 Qxe4 Re8 Nc5 Qxc5 Qg4 Kc7 Qxh3 Be7 Bxg7 Rhf8 Bxf8 Rxf8 Ke1 Qf2 Kd1 Qd4 Kc2 Qe4 Kd2 Bg5 Kc3 Qe5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O Qc2 Ne4 Bxe7 Qxe7 e3 Nxc3 Qxc3 b6 cxd5 exd5 b4 c6 Rc1 Bb7 Bd3 Nd7 Rb1 Qd8 O-O a5 a3 axb4 axb4 g6 Nd2 Qe7 e4 dxe4 Bxe4 Rac8 Rfe1 Qd8 Nc4 Nf6 Bf3 Nd5 Qd2 Ba6 Ne5 Bb5 Rec1 Qd6 h4 Rfd8 h5 Ne7 Re1 Rc7 Bg4 Nd5 hxg6 hxg6 Rb3 f5 Bd1 Rg7 Rh3 Qxb4 Qh6 Qxe1 Kh2 Kf8 Nxg6 Kg8 Qh8 Kf7 Qxd8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O Nb3 Qb6 f3 Rd8 Kb1 a6 Be3 Qc7 Qf2 Nd7 f4 b5 g4 Bb7 Bg2 Rac8 g5 Na5 Nxa5 Qxa5 Ne2 Nc5 Rd4 Qc7 Ng3 d5 exd5 Bxd5 Bxd5 Rxd5 Rxd5 exd5 Rd1 Ne4 Qg2 Qc6 Bd4 Bf8 Nf5 Re8 h4 Qg6 Ne3 Qh5 Rh1 Nc5 Qxd5 Ne4 Ka1 Qf3 Rd1 Qxf4 Qd7 Re6 Rf1 Qxf1 Nxf1 Nc5 Qxe6 Nxe6 Be5 Be7 Ne3 f6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O Nb3 Qb6 f3 a6 Bxf6 Bxf6 Qxd6 Be7 Qg3 Qa7 f4 b5 e5 Bb7 Bd3 Rfd8 f5 Nb4 fxe6 Nxd3 cxd3 fxe6 Kb1 Rf8 Rhf1 Rxf1 Rxf1 Rf8 Rxf8 Bxf8 Ne2 Qb6 Nf4 a5 Nc1 b4 b3 Bc8 Nce2 Qc5 d4 Qc7 Qg4 a4 bxa4 Qc6 Qf3 Qxa4 Qb3 Qc6 d5 exd5 Qxd5 Qxd5 Nxd5 Bb7 Ndf4 g5 Ne6 Be4 Kb2 Bxg2 Nxf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 Bb4 Bd2 Be7 Bg2 d5 Nf3 O-O O-O c6 Qb3 b6 Bf4 Ba6 Nbd2 Nbd7 Rfd1 Nh5 e4 Nxf4 gxf4 Bb7 Rac1 Rc8 Qe3 Nf6 Ne5 Rc7 a3 c5 exd5 exd5 dxc5 Bxc5 Qe2 Qa8 Nb3 dxc4 Bxb7 Qxb7 Rxc4 h6 Rdc1 Rfc8 Nxc5 Rxc5 Rxc5 Rxc5 Rxc5 bxc5 Qc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a3 h6 Bc2 Nb8 b4 Nbd7 Bb2 g6 c4 exd4 cxb5 axb5 Nxd4 c6 a4 bxa4 Bxa4 Qb6 b5 cxb5 Bxb5 d5 Rxa8 Bxa8 Qa4 Nc5 Qa2 Rb8 exd5 Nxd5 Nc4 Qc7 Ne5 Bg7 Nec6 Bxc6 Bxc6 Nf4 Bb5 Rxb5 Nxb5 Qc6 f3 Qxb5 Bxg7 Kxg7 Qa1 Kg8 Qe5 Nfe6 Ra1 Qb7 Qd6 h5 Kh1 Kh7 Rc1 Qa7 Rb1 Ng7 Rb8 Nce6 Qe5 Nd4 Rb1 h4 Qb8 Qe7 Qb4 Qf6 Qf8 Ne2 Rd1 Nf5 Qb8 Ne3 Qd8 Qf4 Re1 Nf1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 e3 h6 Bh4 g5 Bg3 Ne4 Nd2 Nxc3 bxc3 Bxc3 Rc1 Bb4 h4 gxh4 Rxh4 Bd6 Qg4 Bxg3 Qxg3 Nc6 d5 Ne7 Bd3 d6 Qg7 Rg8 Qh7 Rf8 Ne4 Nf5 Rh3 Qe7 g4 Nh4 Qg7 O-O-O Nf6 exd5 cxd5 Kb8 Rxh4 Bxd5 g5 Bxa2 gxh6 d5 h7 Qa3 Rd1 Rh8 Ng8 Bb3 Ra1 Qc5 Qxh8 d4 Rxd4 Qc3 Ke2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Ng5 Rf8 Nf3 Re8 Nbd2 Bf8 a3 h6 Bc2 Nb8 b4 Nbd7 Bb2 g6 c4 exd4 cxb5 axb5 Nxd4 c6 a4 bxa4 Bxa4 Qb6 Nc2 Qc7 Bb3 Ba6 Rc1 Bg7 Ne3 Bb5 Nd5 Nxd5 Bxg7 Kxg7 exd5 Ne5 Ne4 Nd3 Qd2 Ra3 Nf6 Rxe1 Rxe1 Kxf6 Qc3 Ne5 f4 Ba4 fxe5 dxe5 d6 Qxd6 Qf3 Ke7 Qxf7 Kd8 Rd1 Ra1 Qf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 e3 h6 Bh4 g5 Bg3 Ne4 Qc2 Bxc3 bxc3 d6 Bd3 f5 d5 Nc5 h4 g4 Nd4 Qf6 O-O Nba6 Nxe6 Nxe6 Bxf5 Ng7 Bg6 Kd7 f3 Raf8 fxg4 Qe7 e4 Kc8 Qd2 Kb8 Rxf8 Rxf8 Qxh6 Bc8 Re1 Bxg4 c5 Qf6 cxd6 Bh5 e5 Qxg6 Qxg6 Bxg6 e6 Nc5 d7 Nxd7 exd7 Rd8 Re6 Bh5 Be5 Rxd7 Rh6 Bf7 Bxg7 Bxd5 Be5 Bxa2 h5 Kb7 g4 Bc4 g5 a5 g6 Rd5 Bf4 Rf5 Bg3 a4 Rh7 Rc5 h6 a3 Re7 a2 Re1 Rh5 h7 Bd3 Ra1 Bxg6 Rxa2 Rxh7 Kf2 Rd7 Ke2 Rd5 Ra4 c5 Rf4 Be8 Ke3 Rd1 Re4 Bb5 c4 Bd7 Ke2 Rg1 Re7 Rxg3 Rxd7 Ka6 Kd2 Ka5 Rd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 Bg4 dxe5 Qxd1 Rxd1 fxe5 Rd3 Bd6 Nbd2 b5 b3 Ne7 Bb2 Ng6 g3 O-O Kg2 c5 c4 Rab8 a4 bxc4 Nxc4 Bxf3 Rxf3 Rxf3 Kxf3 Rxb3 Ke2 Kf7 Kd2 Ke6 Bc3 Ne7 f4 exf4 Kc2 Rxc3 Kxc3 fxg3 Nxd6 cxd6 hxg3 h5 Rf1 Nc6 Rf8 g5 Rf5 h4 Rxg5 hxg3 Rxg3 Ke5 Re3 Kf4 Rd3 Nd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 e3 h6 Bh4 g5 Bg3 Ne4 Qc2 Bxc3 bxc3 d6 Bd3 f5 d5 Nc5 h4 g4 Nd4 Qf6 O-O Nxd3 Qxd3 e5 Nxf5 Bc8 Nd4 exd4 cxd4 Qf5 e4 Qg6 Qc3 O-O Rfe1 Nd7 e5 Bb7 Re3 b5 Qa5 Nb6 Qxb5 Qc2 exd6 cxd6 Re7 Rf7 Rxf7 Kxf7 c5 Qc4 Qb1 Qxd5 Qh7 Kf6 Qxh6 Kf7 Qf4 Kg8 Qxg4 Kh7 Bf4 Bc8 Qg3 dxc5 Re1 Qf7 Qg5 Nd5 Qh6 Kg8 Re5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 Ne4 Qd3 Qa5 Qxe4 Bxc3 Bd2 Bxd2 Nxd2 Qb6 dxc5 Qxb2 Rb1 Qc3 Qd3 Qxd3 exd3 Na6 d4 Rb8 Bg2 Ke7 Ke2 Rd8 Ne4 b6 Nd6 Nc7 Rb4 Ne8 Nxe8 Kxe8 Rhb1 Ba6 Ke3 d5 cxd6 Rbc8 Kd3 Rxd6 Ra4 b5 cxb5 Rb8 Rab4 Bb7 Bxb7 Rxb7 a4 Ke7 h4 h6 f3 Rd5 Rc1 Rbd7 a5 g5 hxg5 Rxg5 g4 h5 b6 axb6 axb6 Rb7 Rc5 f5 gxh5 Rxh5 Kc4 Rh8 Kb5 Ra8 Rbc4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bf3 Rb8 Qd2 Bd7 Qf2 Nxd4 Bxd4 e5 Be3 Be6 f5 Bc4 Bb6 Qc8 Rfc1 d5 b3 Bb4 Na2 Ba3 bxc4 Bxc1 Nxc1 Qxc4 exd5 e4 Be2 Qxc2 Qd4 Rbc8 h3 e3 d6 Qd2 Nd3 Qxe2 d7 Nxd7 Qxd7 Qd2 Re1 e2 Kg1 a5 g3 Qh6 Bf2 Qc6 Qxc6 Rxc6 Rb1 Rc4 Rxb7 Rxa4 Be1 Ra3 Rd7 a4 Kf2 Rb3 Nc1 Rb1 Na2 Ra8 Re7 Rb2 Rxe2 Rxe2 Kxe2 Re8 Kf2 h5 Bc3 Rb8 Bb4 Rd8 Ke2 a3 Bc3 f6 Bb4 Kf7 Nc3 Rb8 Na2 Rb5 g4 Rb8 Kd3 Rd8 Kc4 Rd1 Bxa3 Ra1 Kb3 Rh1 gxh5 Rxh3 Nc3 Rf3 Bc1 Rxf5 h6 g6 Ne4 Rh5 Bb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 Qb3 c6 e3 Nd7 Rd1 O-O Bd3 b6 cxd5 cxd5 e4 dxe4 Bxe4 Rb8 O-O b5 Rfe1 Qb6 Bb1 Bb7 Qc2 g6 d5 exd5 Nxd5 Bxd5 Rxd5 Rfd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bxf6 Bxf6 e3 O-O Qc2 Na6 Rd1 c5 dxc5 Qa5 cxd5 Nxc5 Qd2 Rd8 Nd4 exd5 Be2 Qb6 O-O Ne4 Qc2 Nxc3 Qxc3 Be6 Qc2 Rac8 Qb1 Rc7 Rd2 Rdc8 Nxe6 fxe6 Bg4 Rc4 h3 Qc6 Qd3 Kh8 Rfd1 a5 b3 Rc3 Qe2 Rf8 Bh5 b5 Bg6 Bd8 Bd3 b4 Qg4 Qe8 e4 Bg5 Rc2 Rxc2 Bxc2 Qc6 Qe2 Qc5 Rf1 Qc3 exd5 exd5 Bb1 Qd2 Qe5 Rd8 Qf5 Kg8 Qe6 Kh8 Qg6 Kg8 Qe6 Kh8 Bf5 Qc3 Qg6 Kg8 Be6 Kh8 Bf5 Kg8 g3 Kf8 Kg2 Qf6 Qh7 Qf7 h4 Bd2 Rd1 Bc3 Rd3 Rd6 Rf3 Ke7 Qh8 d4 Qc8 Rf6 Qc5 Ke8 Rf4 Qb7 Re4 Kf7 Qc4 Kf8 Bh7 Rf7 Qe6 Qd7 Qe5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 Qd7 axb5 axb5 Rxa8 Bxa8 d5 Na5 Ba2 c6 b4 Nb7 c4 Rc8 dxc6 Qxc6 c5 Nd8 Bb2 dxc5 bxc5 Qxc5 Bxe5 Nd7 Bb2 Qb4 Nb3 Nc5 Ba1 Bxe4 Nfd4 Ndb7 Qe2 Nd6 Nxc5 Qxc5 Qg4 Re8 Rd1 Bg6 Qf4 Qb4 Qc1 Be4 Re1 Qa5 Bb3 Qa8 Qb2 b4 Re3 Bg6 Rxe8 Qxe8 Qc1 Ne4 Bd5 Nc5 Nb3 Nd3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bxf6 Bxf6 e3 O-O Qd2 dxc4 Bxc4 Nd7 O-O c5 Rfd1 cxd4 Nxd4 Nb6 Be2 Bd7 Bf3 Rb8 Ne4 Bxd4 Qxd4 Ba4 Qxd8 Rfxd8 Rxd8 Rxd8 Nc5 Rd2 b3 Bc6 Nxb7 Bxf3 gxf3 Nd7 Kg2 g5 b4 Nb6 Kf1 Nd7 Kg2 Nb6 Kf1 Nd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 O-O Bg5 d6 e3 Nbd7 Qc2 b6 Bd3 Bxc3 bxc3 h6 Bh4 Bb7 Nd2 g5 Bg3 Nh5 Qd1 Ng7 h4 f5 hxg5 hxg5 f3 Qe7 Qb3 Kf7 O-O-O Rh8 c5 dxc5 Nc4 cxd4 cxd4 f4 Bf2 Nh5 Bc2 fxe3 Bxe3 Bd5 Qd3 Rag8 Ne5 Nxe5 dxe5 Nf4 Bxf4 gxf4 Rxh8 Rxh8 Qg6 Kf8 Rxd5 exd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qd3 Rfd8 Rfd1 Rab8 Bxc6 Qxc6 Rxb8 Rxb8 dxc5 Bxc3 Qxc3 Qxc5 Qxc5 Nxc5 h3 Ne4 Rxd5 Rb1 Kh2 Nxf2 Rd8 Kh7 Rd7 a5 Rxf7 Rb2 a4 Nd1 Re7 Rb4 Nd4 Nxe3 Nc6 Rc4 Rxe3 Rxc6 Re5 Rc3 Rxa5 Ra3 h4 Kg6 g4 Kf6 Rf5 Ke6 Rf4 g5 hxg5 hxg5 Rb4 Ke5 Kg2 Ra2 Kf3 Ra3 Ke2 Rg3 Kd2 Rg2 Ke3 Rg3 Ke2 Ra3 Kd2 Rg3 Rc4 Kd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 Nb8 Bd3 c6 Nf1 Nbd7 Ng3 Qc7 Bd2 g6 Qc1 Kh7 b3 Bg7 Qc2 Nf8 Be3 Ne6 Rad1 Rac8 Bf1 Bf8 Rd2 Qb8 Qb1 Ba8 b4 Bb7 axb5 axb5 Red1 Qc7 Rc1 Bg7 Rcd1 Rcd8 dxe5 dxe5 Rxd8 Rxd8 Rxd8 Nxd8 c4 bxc4 Bxc4 Ne8 Qa2 Nd6 Bb3 Nb5 h4 Nd4 Bxd4 exd4 h5 Qe7 Qd2 c5 Qc2 cxb4 hxg6 fxg6 Qc4 h5 e5 Bxf3 gxf3 Bxe5 f4 Bxf4 Qg8 Kh6 Bc2 Qg7 Qxd8 Bxg3 fxg3 Qe5 Qf8 Kg5 Kg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bg1 Rb8 Qd2 e5 Nb3 Na5 Nxa5 Qxa5 Ba7 Ra8 Be3 Qb4 Qd3 Be6 f5 Bd7 Ra3 Qa5 Rb3 b5 axb5 axb5 Nxb5 Bc6 Bf3 Rab8 c4 Qa8 Bg5 Bxe4 Bxe4 Nxe4 Bxe7 Rxe7 Ra3 Qc6 b4 h5 Na7 Rxa7 Rxa7 Rxb4 Qf3 Rxc4 Qxh5 Nf2 Kg1 Nh3 Kh1 Nf2 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 O-O Bg5 c5 e3 cxd4 exd4 h6 Bh4 d5 Rc1 dxc4 Bxc4 Nc6 O-O Be7 Re1 b6 a3 Bb7 Bg3 Rc8 Ba2 Bd6 d5 Nxd5 Nxd5 Bxg3 hxg3 exd5 Bxd5 Qf6 Qa4 Rfd8 Rcd1 Rd7 Qg4 Rcd8 Qxd7 Rxd7 Re8 Kh7 Be4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 d5 exd5 exd5 cxd5 Nb4 Bc4 Bg4 Be2 Bxe2 Qxe2 Qe7 Be3 Nbxd5 Nc2 Nxe3 Nxe3 Qe6 O-O Bc5 Rfe1 O-O
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 Nc6 Bg2 Ne4 Bd2 Bxc3 bxc3 O-O O-O f5 Be3 Nxc3 Qd3 cxd4 Nxd4 Ne4 c5 Nxd4 Bxd4 b6 Bxe4 fxe4 Qxe4 Ba6 cxb6 axb6 Qe5 Qf6 Qe3 Qh6 Qxh6 gxh6 Rfe1 Bc4 a3 b5 Rad1 Rf5 Bb2 Rd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 g4 h6 h4 a6 Bg2 Be7 Be3 Nxd4 Qxd4 e5 Qd1 Be6 Nd5 Rc8 c3 Nf6 Nxe7 Qxe7 g5 hxg5 hxg5 Rxh1 Bxh1 Ng4 Bd2 Qf8 Qf3 Qh8 Bg2 Qh4 b3 d5 Qg3 Qxg3 fxg3 Rd8 Ke2 Ke7 Bc1 d4 Ba3 Ke8 cxd4 exd4 Rh1 Ne5 Rh8 Kd7 Rxd8 Kxd8 Bb2 Bg4 Kd2 Nf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Bg4 c4 Nf6 Nc3 Bxf3 Qxf3 Nxd4 Re1 Be7 Qd1 Ne6 cxd5 Nxd5 Bb5 c6 Nxd5 cxb5 Qb3 O-O Nxe7 Qxe7 Qxb5 a6 Qb3 Rfd8 Be3 Rac8 Rac1 h6 h3 Nd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 d5 cxd5 exd5 exd5 Nb4 Be2 Bc5 O-O O-O Bf3 Bf5 Bg5 Re8 Qd2 b5 Rad1 Nd3 Nab1 h6 Bh4 b4 Na4 Bd6 Bg3 Rc8 b3 g5 Bxd6 Qxd6 g3 Nd7 Bg2 Qf6 a3 a5 axb4 axb4 Qa2 Bg6 d6 g4 Qd2 Kg7 f3 Qxd6 fxg4 Qd4 Kh1 Nf6 Rf4 Ne4 Qxd3 Nf2 Rxf2 Bxd3 Rfd2 Qe3 Rxd3 Rc1 Nb2 Qf2 Nd2 Rxd1 Nxd1 Re1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 Nc6 Bg2 Ne4 Bd2 Bxc3 bxc3 O-O O-O Na5 dxc5 Qc7 Nd4 Nxd2 Qxd2 Nxc4 Qg5 f6 Qf4 Ne5 Nb3 Rb8 Qd4 b6 f4 Nf7 Rfd1 Rd8 c4 Bb7 Bxb7 Rxb7 cxb6 Rxb6 c5 Rc6 Rac1 d5 cxd6 Rdxd6 Qe3 Rxd1 Rxd1 g6 Rc1 Rxc1 Qxc1 Qb6 Qc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bf3 Rb8 Qd2 Bd7 Nb3 b6 Bf2 Bc8 Bg3 Nd7 Rae1 Bb7 e5 Rbd8 Qf2 Rf8 Be4 dxe5 fxe5 Nc5 Nxc5 bxc5 Bf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 Ne4 Qc2 f5 g3 Nc6 Bg2 O-O O-O Bxc3 bxc3 Na5 c5 d6 c4 b6 Bd2 Nxd2 Nxd2 d5 cxd5 exd5 e3 Be6 Qc3 Rf7 Rfc1 Rb8 Rab1 Re7 a4 Bf7 Bf1 h6 Bd3 Qd7 Qc2 Be6 Bb5 Qd8 Rd1 g5 Nf3 Rg7 Ne5 f4 Bf1 Qf6 Bg2 Rd8 e4 dxe4 Bxe4 Re7 Qc3 Bd5 Re1 Kg7 Ng4 Qf7 Bxd5 Rxd5 Rxe7 Qxe7 Re1 Qd8 Ne5 Qf6 cxb6 Qxb6 gxf4 Rxd4 Nf3 Nb3 Rb1 Qf6 Qxc7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 Qc2 O-O e3 c5 dxc5 Bxc5 Nf3 Nc6 Be2 d4 exd4 Nxd4 Nxd4 Qxd4 Bg3 Be6 O-O Rac8 Bf3 b6 Rfe1 Qb4 Be5 Bd4 a3 Qc5 Bxd4 Qxd4 Rad1 Qc5 Qa4 a5 Qd4 Qxd4 Rxd4 Rfd8 Red1 Rxd4 Rxd4 Kf8 Kf1 Ke7 Ke2 Bb3 Ke3 Rc5 Kd2 h6 Be2 Ne8 Bf3 Nf6 Rd3 Re5 h3 Rc5 Rd4 Rc8 Be2 Rc5 Bd3 h5 g3 g6 Ne2 Nd7 Re4 Re5 Nd4 Bd5 Re2 Rxe2 Bxe2 Nc5 Nb5 Ne4 Ke3 Nd6 Kd4 Bc6 Nxd6 Kxd6 Bc4 Be8 h4 f6 Bg8 Kc6 Ba2 Kd6 Bd5 Ke7 Bg8 Kd6 Bb3 Ke7 Bd1 Kd6 Be2 Bd7 Bd3 Be8 Bc4 Ke7 Be2 Kd6 g4 hxg4 Bxg4 Bf7 f4 f5 Bd1 Bd5 Ba4 Bf3 Bb3 Be2 Bf7 Bh5 Kc4 Be2 Kc3 Bh5 b4 Ke7 Bc4 Kd6 bxa5 bxa5 Kd4 Bf3 Bf1 Bd5 Be2 Bb7 Bd1 Bd5 Ke3 Kc5 Ba4 Bf7 Bd7 Kc4 a4 Kc5 Bb5 Kd5 Kd3 Kc5 Kc3 Kd6 Kd4 Bb3 Be8 Ke7 Bxg6 Bxa4 Bxf5 Kf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 c6 e3 Bf5 g4 Be6 h4 Nd7 h5 Nh6 Be2 Nb6 Rc1 Bd6 Nh3 Bxf4 Nxf4 Bd7 Rg1 g5 hxg6 hxg6 Kd2 Qe7 b3 g5 Nd3 O-O-O Rh1 f6 Qg1 Nf7 Qg3 Qd6 Qxd6 Nxd6 f3 Rdg8 Nc5 Kd8 Bd3 Bc8 Ne2 Na8 Bh7 Rf8 Rh6 Nc7 Ng3 Nf7 Rh2 Ne6 Nd3 Ng7 Rch1 Ke7 Nf2 Rd8 Bf5 Rxh2 Rxh2 Nxf5 gxf5 Rh8 Rxh8 Nxh8 e4 Nf7 Ng4 Nd6 Ne3 dxe4 fxe4 b6 b4 Ba6 Ng4 Nb5 Kd3 Na3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 e3 O-O Nf3 Bf5 h3 c6 g4 Bg6 Ne5 Nfd7 Nxg6 fxg6 Bg2 Nb6 O-O Kh8 Ne2 g5 Bg3 Bd6 Qd3 Na6 b3 Qe7 Bxd6 Qxd6 f4 gxf4 exf4 Rae8 f5 Nc7 Rf2 Nd7 g5 Qe7 h4 Qe3 Rd1 Nb5 Qxe3 Rxe3 Kh2 Nb6 Ng3 Nc8 Nf1 Re7 Rd3 Ncd6 Ng3 Ne4 Bxe4 dxe4 Re3 Nxd4 Kh3 Re5 Kg4 h5 Kxh5 Nxf5 Rxf5 Rfxf5 Nxf5 Rxf5 Rxe4 Kh7 Re7 b5 Rxa7 b4 Kg4
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 e5 h3 exd4 exd4 Nb6 Bb3 Re8 Re1 Bf5 Rxe8 Qxe8 Qd2 Qd7 Re1 Rd8 Qf4 Nd5 Nxd5 cxd5 Ne5 Bxe5 Rxe5 Be6 Qe3 Kf8 Qd3 f6 Re1 Bf7 Qc3 Qd6 Rc1 Be8 Bd1 a6 Bf3 g6 h4 h5 g3 Bf7 a4 Rd7 a5 Kg7 Qb3 Qe6 Qb4 Qe8 Kg2 Qd8 Rc5 Qe7 Qc3 g5 Qe3 g4 Bd1 Qe4 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bf3 Rb8 Qd2 Bd7 Nb3 b6 g4 Bc8 g5 Nd7 Qf2 Bf8 Bg2 Bb7 Rad1 g6 Bc1 Rbc8 Rd3 Nb4 Rh3 Bg7 Be3 Re7 Kg1 Rce8 Rd1 f5 gxf6 Nxf6 Rg3 Rf7 Bxb6 Qb8 Be3 Nh5 Rg4 Nf6 Rh4 g5 fxg5 Ng4 Qd2 Nxe3 Qxe3 Nxc2 Qb6 Ba8 Rxd6 Rb7 Qxa6 Rxb3 Rxe6 Rxb2 Qc4 Kh8 e5 Qa7 Kh1 Bxg2 Kxg2 Nd4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 e5 Nc3 d6 e4 Be7 Nf3 O-O h3 Nbd7 g4 Ne8 Bd3 a6 a4 Rb8 Rg1 Nc7 b3 Re8 h4 b5 g5 Nf8 h5 Bd7 Nh2 bxc4 Bxc4 f5 exf5 Bxf5 Nf1 Qd7 Ne3 e4 Bb2 Bd8 Ne2 Qf7 Nf4 Bc8 Rg4 Qe7 Rg3 Qf7 Nfg2 Na8 a5 Nc7 Nh4 Nb5 g6 hxg6 Nxg6 Bf6 Bxb5 Rxb5 Qc2 Bxb2 Qxb2 Nxg6 Rxg6 Re5 O-O-O Rxh5 Rdg1 Rh7 Nc4 Qf4 Kb1 Rb7 Nxd6 Bf5 Rf6 Qh2 Rg3 Qh1 Ka2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Qc2 g6 Bf4 dxc4 Qxc4 Bg7 Nc3 O-O e4 b5 Qb3 Qa5 Bd3 Be6 Qd1 Rd8 O-O Bg4 e5 Nd5 Nxd5 cxd5 Rc1 Qb6 Rc5 Nd7 Rxb5 Bxf3 Qa4 Bxg2 Rxb6 Nxb6 Qa6 Bxf1 Kxf1 e6 b3 Rd7 a4 Rc7 Bb5 Bf8 Bg5 Rb8 Qa5 Rbc8 Kg2 Ba3 Qe1 Bb2 Ba6 Bc3 Bd2 Bxd2 Qxd2 Rc2 Qg5 R8c7 Qd8 Kg7 Qf6 Kg8 Qd8 Kg7 a5 Nc8 Qf6 Kg8 Qd8 Kg7 Bd3 R2c3 Qf6 Kg8 Qd8 Kg7 Bc4 Rb7 Ba6 Rbc7 b4 Ne7 b5 Nf5 b6 axb6 axb6 R7c6 Bb5 Rc8 Qf6 Kg8 Bd7 Rb8 Bxe6 fxe6 Qxe6 Kg7 Qf6 Kh6 h4 Rb3 Qg5 Kg7 Qf6 Kg8 Qe6 Kh8 Qxd5 Nxh4 Kh2 R3xb6 e6 Nf5 Qe5 Kg8 Qc7 R6b7 Qa5 Rb2 Kg1 Re2 d5 Rb1 Kg2 Rbb2 Qd8 Kg7 Qc7 Kh6 Qf4 Kh5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c5 d5 b5 Bg5 Ne4 Bh4 Qa5 Nbd2 Bb7 a4 Bxd5 axb5 Qc7 Ra4 Qb7 c4 Nxd2 cxd5 Nxf1 Qd3 d6 e4 Nd7 Qxf1 h6 Qe2 g5 Bg3 Bg7 e5 O-O h4 Qxd5 hxg5 Nxe5 Bxe5 dxe5 gxh6 Bf6 Rh5 Kh8 Nxe5 Qb3 Ra3 Qb4 Kf1 Rad8 Nc6 Qxb2 Qxb2 Bxb2 Rxa7 Rc8 Rxe7 c4 Ke2 c3 Kd3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6 e4 c5 e5 cxd4 Nxb5 Ng4 Nbxd4 Bb4 Bd2 Bxd2 Qxd2 Bb7 Rd1 O-O O-O Ngxe5 Be2 Qb6 b3 Nxf3 Bxf3 Nf6 Bxb7 Qxb7 Nc2 a5 Ne3 Rfb8 Qb2 Qc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Qa5 g3 b4 Ne4 Ba6 Qf3 O-O-O b3 cxb3 Bxa6 Qxa6 Qxb3 Qb5 Rc1 Nb6 Be3 a5 Qc2 Kb7 Qe2 Qd5 f3 Nd7 O-O Bh6 Rf2 Kb6 a3 Rb8 axb4 axb4 Bxh6 Rxh6 Qd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6 e4 c5 e5 cxd4 Nxb5 Ng4 Qa4 Ngxe5 Nxe5 Nxe5 Nd6 Ke7 Nxc8 Kf6 Be4 Rxc8 h4 h6 O-O Rc4 Qd1 d3 Re1 Rxc1 Rxc1 d2 Rf1 Qd4 Rc2 Qxe4 Rxd2 Bc5 Re1 Qxh4 Qc2 Bb4 Rxe5 Bxd2 g3 Qd4 Re4 Qd5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd3 Qxb2 Rb1 Qa3 f5 Be7 Be2 Nc6 fxe6 fxe6 Nxc6 bxc6 e5 dxe5 Bxf6 gxf6 Bh5 Kf8 Qd2 Kg7 Rb3 Qa5 O-O Rg8 Kh1 Kh8 Qh6 Qd8 Ne4 f5 Bf7 Qf8 Qh5 Rg7 Be8 a5 Bxc6 Ba6 Rf2 Rd8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 Qd2 b5 f3 Nbd7 g4 h6 O-O-O Bb7 Bd3 Ne5 Rhe1 Rc8 Kb1 Be7 h4 b4 Na4 Qa5 b3 Nfd7 g5 g6 f4 Nxd3 cxd3 hxg5 hxg5 d5 f5 e5 exd5 Qxd5 f6 Bd6 Nc2 a5 Ba7 Kf8 Ne3 Qe6 Nc4 Kg8 Nxd6 Qxd6 Nb2 Rc3 Nc4 Qd5 Ne3 Qe6 Rc1 Qa6 Rxc3 bxc3 Qxc3 Qxa7 Qc7 Qd4 Qxb7 Qxd3 Nc2 Rh2 Qc8 Nf8 Rxe5 Rh1 Kb2 Qd2 Re8 Qd6 Rd8 Qe5 Ka3 Kh7 Rxf8 Qd6 b4
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 Nf3 Nf6 Bg5 h6 Bh4 O-O e3 b6 Rc1 Bb7 Bd3 dxc4 Bxc4 Nbd7 O-O c5 Qe2 a6 a4 cxd4 exd4 Nh5 Bxe7 Qxe7 d5 Nf4 Qe3 Qf6 Ne4 Qf5 Ng3 Qf6 Ne4 Qf5 Ng3 Qf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qa4 Bd7 Qb3 dxc4 Qxc4 O-O e4 b5 Qb3 c5 e5 Ng4 Bxb5 cxd4 Nxd4 Bxb5 Ndxb5 a6 Na3 Qd4 Qc2 Nc6 Qe2 Qxe5 Qxe5 Ngxe5 O-O Nd3 Rb1 Rab8 Rd1 Rfd8 Kf1 f5 Ke2 Nce5 Na4 Rd6 Be3 f4 Bc5 f3 gxf3 Nf4 Ke3 Rf6 Bxe7 Ng2 Ke2 Rxf3 Bd6 Nf4 Kf1 Ng4 Rd2 Re8 Nc4 Nxh2 Kg1 Ng4 Rf1 Bd4 Bc5 Rg3 Kh1 Rh3 Kg1 Nh2
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Bf4 Nf6 e3 O-O Rc1 c5 dxc5 Bxc5 cxd5 exd5 Nf3 Nc6 Be2 Be6 O-O Re8 Nb5 Ne4 Nd2 Nxd2 Qxd2 Bb4 Qd1 Qb6 a3 Be7 b4 a5 Nd6 Red8 Nxb7 Qxb7 b5 Rac8 bxc6 Rxc6 Rb1 Rb6 Rxb6 Qxb6 Be5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 Nbd7 Qc2 h6 cxd5 exd5 Bf4 c5 Be2 b6 O-O Bb7 Rfd1 Re8 dxc5 bxc5 a4 Qa5 Nh4 Red8 Nf5 Bf8 Nb5 Ne8 Bd6 Nxd6 Nfxd6 Rab8 Nxb7 Rxb7 Rxd5 Rdb8 Qd2 Qxd2 Rxd2 Nf6 Ra2 Ne4 Rc2 Rd7 g3 a5 Kg2 g6 Bf3 Nf6 Na3 Bd6 Bc6 Rdd8 Ra1 Be5 Bb5 Nd5 Rb1 Bd6 Rd2 Nb6 Rc1 Be7 Re2 Rbc8 Nb1 Kg7 Nd2 Ra8 Nb3 Rdc8 Rec2 c4 Nd2 Ra7 Nxc4 Nxc4 Rxc4 Rxc4 Rxc4 f5 h3 h5 g4 hxg4 hxg4 fxg4 Kg3 Bd6 Kxg4 Re7 Bc6 Rf7 f4 Kh6 Bd5 Rf6 Rc1 Kg7 b3 Rf8 Rd1 Bc5 Rd3 Ba3 Be4 Bc1 Rd7 Kh6 Re7 Bd2 Kf3 Bb4 Rb7 Bc3 Bd3 Rf6 Kg4 Bd2 f5
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 Nf3 Nf6 c4 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 e5 h3 exd4 exd4 Nb6 Bb3 Bf5 Re1 Bg5 Ra1 Nd7 d5 Rc8 Nd4 Bg6 Ne6 fxe6 dxe6 Kh7 Qxd7 Qb6 e7 Rfe8 Qg4 Qc5 Ne4 Qxe7 Bc2 Rf8 g3 Qd8 Rad1 Qa5 h4 Be7 Nc3 Bxc2 Rxe7 Rg8 Rdd7 Bf5 Rxg7 Kh8 Qd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 Bb4 a3 Bd6 e4 dxe4 Nxe4 Nxe4 Bxe4 e5 O-O O-O Bc2 Re8 Re1 exd4 Rxe8 Qxe8 Qxd4 Be7 Bg5 Bxg5 Nxg5 Nf6 Rd1 Be6 Re1 Qd8 Nxe6 fxe6 Qe3 Kh8 h3 Qd7 g4 Re8 Qe5 Qd8 Kg2 Qb6 Rd1 c5 Ba4 Rf8 Rd6 Qc7 Rxe6 Qf7 Qxc5 Nxg4 Qxf8 Qxf8 hxg4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 c5 g3 Qb6 Bg2 Nc6 d5 exd5 cxd5 Nxd5 O-O Nde7 e4 d6 Be3 Qc7 a3 Ba5 Bf4 Ne5 b4 cxb4 axb4 Bxb4 Qa4 N7c6 Nd4 a5 Nc3 Bd7 Nd5 Qd8 Nf5 O-O Qd1 Bc5 Rc1 a4 g4 a3 g5 a2 Qh5 Bxf5 exf5 Bd4 Bxe5 Nxe5 Be4 Re8 Rc7 a1=Q Rxa1 Rxa1 Kg2 Ra2 Re7 Rxf2 Kg3 Rxe7 f6 Ng6 Nxe7 Kf8 Qxh7 Bxf6 Nxg6 fxg6 Kxf2 Qb6 Kg2 Qb2 Kh3 Bxg5 Qxg6 Qf6 Qxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 c6 Nc3 d5 cxd5 cxd5 Ne5 e6 O-O Nfd7 f4 Nxe5 fxe5 Nc6 e4 dxe4 Be3 f5 exf6 Rxf6 Nxe4 Rxf1 Qxf1 Nxd4 Rd1 e5 Ng5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 Qd7 axb5 axb5 Rxa8 Bxa8 d5 Nb8 Nf1 Na6 Bg5 Be7 Ng3 g6 Qd2 Bb7 Ra1 Ra8 Bc2 c6 dxc6 Bxc6 Rd1 Rd8 Qe3 Qb7 Bh6 Nc7 Nf5 Ne6 Nxe7 Qxe7 Ng5 Nc5 b4 Na4 Bb3 Be8 Rd2 Rc8 Rc2 Nd7 Nf3 Ndb6 Bg5 Qc7 Nd2 Kg7 c4 h5 cxb5 Qd7 Qf3 Rxc2 Bxc2 Kg8 Bb3 Qxb5 Qf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 e6 Nc3 c6 e3 Nbd7 Qc2 Bd6 b3 O-O Be2 a6 O-O e5 cxd5 cxd5 dxe5 Nxe5 Bb2 Be6 Rac1 Rc8 Qb1 Nfg4 Nxe5 Bxe5 Bxg4 Bxg4 Na4 Bb8 Rxc8 Bxc8 Bd4 Re8 Nc3 Qd6 f4 f6 b4 Be6 Na4 Bd7 Nb6 Bc6 a4 Bc7 a5 Re4 Qb3 Qe6 Qc3 Bd6 h3 h6 Qb3 Kh8 Na4 Bb5 Rf3 Rxd4 exd4 Qe1 Kh2 Qe4 Nc5 Qxd4 Ne6 Qe4 Re3 Bxf4 Nxf4 Qxf4 Kg1 d4 Rf3 Qd6 Rf2 Qe5 Qd1 Bc6 Qd2 Be4 Qb2 Qd5 Qc1 f5 Qc5 Qd7 Qc1 d3 Qe3 Qd5 Qc5 Qd7 Rd2 Kh7 Kf2 Qd8 Ke1 Qg5 Qf2 h5 g3 Qf6 Qf4 Qd4 Kf1 Qa1 Kf2 Qh1 h4 Qg2 Ke3 Qg1 Qf2 Qb1 Rb2 Qc1 Kd4 Kg8 Qd2 Qc6 Qg5 Qd6 Ke3 Qe5 Rd2 Qd6 Rb2 Qc7 Kd2 Qe5 Rb1 Qd5 Qxh5 Qa2 Ke3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 c5 Qe2 a6 Rfd1 cxd4 Nxd4 Qe7 Ne4 Be5 Qh5 Rd8 Bf1 Bb8 Qa5 b6 Qc3 Bb7 Nc6 Bxc6 Qxc6 Ra7 g3 b5 b3 Rf8 Be2 Nf6 Nxf6 Qxf6 Qb6 Qe7 Rc6 Be5 Bf3 Rb8 Qa5 Bc7 Qc3 Rb6 h4 g6 Rc5 Rb8 h5 g5 Rc6 Bb6 Kg2 b4 Qe5 Rc7 Rdd6 Rxc6 Rxc6 Qd8 Bg4 Rb7 Bxe6 fxe6 Qxe6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bg5 Ne4 cxd5 Nxg5 Nxg5 e6 Nf3 exd5 b4 Qd6 a3 O-O e3 c6 Be2 Bf5 O-O Nd7 Na4 a5 Qb3 b5 Nc5 a4 Qc3 Nb6 Nd2 Rae8 Rfe1 Re7 Bf3 Rfe8 g3 Bh3 Bg2 Bxg2 Kxg2 f5 h4 Nc4 Nf3 Bf6 Re2 Rg7 Rh1 Qe7 Ree1 h6 Qd3 Rf8 Nd2 Qe8 Nxc4 dxc4 Qd1 Re7 Ref1 Qf7 Qf3 Qd5 Qxd5 cxd5 Kf3 Bg7 Rd1 Rff7 Rd2 Re8 Rdd1 Bf8 Rdg1 Bg7 Rd1 Kf8 Rd2 Ke7 Rdd1 Kd6 Rh2 Kc6 Rhh1 Bf8 Rd2 Bd6 Rdd1 Bxc5 dxc5 Re4 Rhe1 Rd7 Rd4 g5 hxg5 hxg5 Red1 Rxd4 Rxd4 Rh7 Ke2 Rh3 g4 f4 exf4 Rxa3 fxg5 Ra2 Kf3 c3 Rd1 d4 g6 d3 Ke3 Rxf2 g7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 cxd4 Nxd4 O-O Bg2 d5 Qb3 Bxc3 bxc3 Nc6 cxd5 Na5 Qc2 Nxd5 Qd3 Qc7 O-O Bd7 e4 Nb6 f4 e5 fxe5 Nac4 g4 Nxe5 Qg3 Nbc4 Bf4 Qc5 Rae1 Rad8 g5 Rfe8 Kh1 b5 Rd1 Bc8 h4 Bg4 Nb3 Qc8 Rd5 Rxd5 exd5 Bh5 Nd4 Ng6 Bc1 Qd7 Nf5 Re5 Qf2 Be2 Re1 Rxf5 Qxe2 Re5 Qf2 Rxe1 Qxe1 Qg4 Kh2 Qxh4 Qxh4 Nxh4 Bf1 Nf5 a4 a6 axb5 axb5 Bd3 Nfd6 Bf4 Kf8 Kg3 Ke7 Bc1 Ne5 Bc2 Ndc4 Kf4 Kd6 Ke4 Kc5 Bb3 Ng6 Bxc4 Kxc4 Be3 Nf8 Bd4 h5 Bxg7 Nd7 d6 Kc5 Be5 h4 Bf4 Kc4 Bd2 h3 Kf3 Kd5 Be1 Ne5 Kg3 Ke4 c4 bxc4 Bc3 Nd7 Kxh3 Kd3 Bb2 Ke4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 g3 Bc5 Nb3 Be7 Bg2 Nf6 O-O d6 c4 Ne5 Qe2 Qc7 Na3 Bd7 Bf4 Rc8 Rac1 b6 Nb5 Qb8 Rfd1 O-O N5d4 Rfd8 Bd2 Ba4 Re1 a6 Na1 Be8 b3 Nfd7 f4 Ng6 Nac2 Bf6 Kh1 Nc5 Bc3 Bd7 Ne3 b5 Qd2 bxc4 Nxc4 d5 exd5 exd5 Ne5 Bxe5 fxe5 Ne4 Bxe4 dxe4 Ba5 Re8 e6 fxe6 Nc2 Ne5 Ne3 Qb5 Qb4 Rxc1 Rxc1 Qe2 Qd2 Qf3 Kg1 Qh5 Rc5 Nf3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Bb4 cxd5 exd5 Bg5 Nbd7 e3 c5 Bd3 Qa5 Qc2 O-O O-O c4 Bf5 Re8 Nd2 g6 Bxd7 Nxd7 a3 Bxc3 bxc3 Nb6 f3 Bd7 Rfe1 Ba4 Qc1 Bc6 Ra2 Re6 Nf1 Rae8 h4 Nc8 Nh2 f6 Bf4 Nd6 Bxd6 Rxd6 h5 Qc7 hxg6 hxg6 Ng4 Qg7 Rae2 Bd7 Nf2 Qh6 Qc2 Rde6 e4 dxe4 Nxe4 Kg7 d5 Re5 Nd6 Rxe2 Rxe2 Rxe2 Qxe2 Qc1 Kf2 Kf8 Qe3 Qxa3 Qh6 Ke7 Qg7 Kd8 Qf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 a4 b4 Nd2 g6 e4 d6 Ngf3 Bg7 g3 e6 Bh3 exd5 Bxc8 Qxc8 cxd5 O-O O-O c4 Qc2 c3 bxc3 bxc3 Nb3 Qg4 Nfd4 Qxe4 Qxc3 Nxd5 Qd2 Nb6 Re1 Qd5 Qd1 N8d7 Ra2 Bxd4 Rd2 Ne5 Nxd4 Nbc4 Rc2 Rac8 Rc3 Nb6 Bb2 Rxc3 Bxc3 Rc8 Ba1 Rc4 Nb5 Rxa4 Qxd5 Nxd5 Bxe5 dxe5 Rxe5 Nf6 Re7 a5 Nd6 Rd4 Nxf7 Kf8 Ra7 Rd7 Rxd7 Nxd7 Nd6 Ke7 Nc4 a4 Kf1 Ke6 Ke2 Kd5 Ne3 Kd4 Kd2 Ne5 Kc2 Nd3 Nd1 Ne1 Kb2 Nf3 h4 Ne5 Ka3 Ke4 h5 gxh5 Nc3 Kf3 Nd5 Ng4 Ne7 Nxf2 Nf5 Ne4 Kxa4 Nxg3 Nh4 Ke4 Kb4 Nf5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bf4 c5 dxc5 Qa5 Rc1 dxc4 e3 Qxc5 Qa4 Nc6 Bxc4 O-O O-O Bd7 Qb5 Qxb5 Bxb5 Rac8 Rfd1 Rfd8 h3 h6 Kf1 a6 Be2 Be6 Rxd8 Rxd8 Ne5 Nxe5 Bxe5 Rd2 b3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 Nc6 Bg2 d5 cxd5 Nxd5 Bd2 cxd4 Nxd4 Nxd4 Nxd5 Bxd2 Qxd2 Nc6 Nf4 Qxd2 Kxd2 Bd7 Rhc1 Ke7 Nd3 Rhc8 Nc5 Rab8 Rc3 Nd8 Rac1 Bc6 Nd3 Bd7 Ne5 Rxc3 Rxc3 Be8 b4 a6 Be4 h6 a3 f6 Nd3 Bc6 Bxc6 Nxc6 Nc5 Ne5 f4 Nd7 Nb3 Kd6 e4 g5 Ke3 e5 fxg5 fxg5 Na5 g4 Rc2 h5 Rc1 b6 Rc6 Ke7 Nc4 Rf8 Ke2 Rf3 Ne3 Nf6 Rxb6 Nxe4 Rxa6 Rf2 Kd3 Nd6 Ra7 Ke6 Rh7 e4 Kc3 Nb5 Kc4 Nxa3 Kd4 Rxh2 Rh6 Kd7 Nd5 h4 Rxh4 Rxh4 gxh4 g3 Nf4 Nc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 c6 Bg2 d5 cxd5 cxd5 Nc3 O-O Ne5 e6 O-O Nfd7 Nf3 Nc6 Bf4 Nf6 Ne5 Bd7 Qd2 Nxe5 Bxe5 Bc6 Rfd1 Nd7 Bxg7 Kxg7 Rac1 Nf6 Qf4 Qb8 Qxb8 Raxb8 f3 Rfd8 Kf2 Rbc8 e3 Ne8 Rd2 Nd6 Rdc2 Kf8 Bf1 Ke7 Bd3 f5 h4 h6 b3 g5 Ne2 Bd7 Rc5 b6 Rc7 Rxc7 Rxc7 Ra8 Ng1 Ne8 Rc1 Rc8 Rxc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 cxd4 Nxd4 O-O Bg2 d5 Qb3 Bxc3 bxc3 Nc6 cxd5 Na5 Qc2 Nxd5 Qd3 Bd7 c4 Ne7 O-O Rc8 Nb3 Nxc4 Bxb7 Rc7 Ba6 Ne5 Qe3 Nc4 Qe4 Nd6 Qd3 Rc6 Ba3 Bc8 Bxc8 Ndxc8 Rfd1 Qxd3 Rxd3 Re8 Rad1 f6 Nd4 Rb6 Bc5 Ra6 Nb5 Rc6 Bxe7 Nxe7 Rd7 Ng6 Rxa7 Nf8 a4 Rb8 e3 h5 Kg2 e5 Rd3 Kh7 Rc3 Rbc8 Rxc6 Rxc6 Nc7 Ne6 Nd5 Kh6 a5 e4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Rc1 Ne4 cxd5 Nxc3 Qd2 Qxa2 bxc3 Qxd2 Kxd2 Nd7 Bb5 O-O Bxd7 Bxd7 e4 f5 e5 e6 c4 Rfc8 c6 bxc6 d6 c5 h4 h6 Nh3 a5 f3 a4 Rhe1 a3 Nf2 a2 Nd3 Ra3 Ra1 g5 hxg5 hxg5 Bxg5 Kf7 Bf4 Rb8 Rec1 Bc6 Rc3 Ra5 Rc2 Rba8 Nc1
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Bg4 c4 Nf6 Nc3 Bxf3 Qxf3 Nxd4 Qe3 Ne6 cxd5 Nxd5 Nxd5 Qxd5 Be4 Qb5 a4 Qa6 Rd1 Be7 b4 O-O Qh3 g6 Bb2 Qc4 Rd7 Rae8 Bd5 Qxb4 Bc3 Nf4 Bxb4 Nxh3 gxh3 Bxb4 Rxc7 b6 Rxa7 Kg7 Rd7 Rd8 Rxd8 Rxd8 Rd1 Rd6 Rd3 h5 Kf1 Rd7 Kg2 Bc5 Kf1 h4 Bc4 Re7 Rf3 Bd6 Kg2 Rc7 Bb3 f5 Rd3 Bc5 Rc3 Kf6 Rc4 g5 Rc2 Ke5 Bc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 c6 Qc2 g6 e3 Bf5 Qd2 Nd7 f3 Nb6 e4 Be6 e5 h5 Bd3 Qd7 b3 Bh4 g3 Be7 Kf2 Bf5 Bf1 Kf8 Kg2 a5 a3 Qd8 Nh3 Bxh3 Kxh3 Kg7 Kg2 Nd7 Bd3 Nf8 Be3 Ne6 Ne2 Nh6 b4 Qb6 b5 c5 Nc3 cxd4 Bxh6 Rxh6 Nxd5 Qd8 Be4 h4 Rhf1 hxg3 hxg3 Rc8 Rh1 Rxh1 Rxh1 Bg5 f4 Rc5 fxg5 Rxd5 Bxd5 Qxd5 Kh2 Qxe5 Rf1 Qxb5 Qf2 Nxg5 Qxd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 e3 O-O Bd3 c5 Nf3 Nc6 O-O Bg4 dxc5 Bxc5 h3 Bxf3 Qxf3 d4 Ne4 Be7 Rad1 Qa5 Ng3 dxe3 fxe3 Qxa2 Nf5 Qe6 Bh6 Ne8 Qh5 g6 Qg4 Ne5 Qg3 Bf6 Bb5 Ng7 Bxg7 Bxg7 Rd6 Qb3 Nxg7 Qxb5 Nf5 Rad8 Rf6 Rd2 Qg5 Qxb2 Kh1 Kh8 Nd4 Rxd4 Qxe5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Rc1 dxc4 Bxc4 O-O Nf3 Qxc5 Bb3 Nc6 O-O Qa5 h3 Bf5 Nd4 Bd7 Qe2 Nxd4 exd4 e6 Bd2 Qb6 Rfd1 Bc6 Be3 Qa5 Bd2 Qb6 Be3 Qa5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 e5 h3 exd4 exd4 c5 Bb3 cxd4 Nd5 b6 Nxd4 Bxd4 Qxd4 Nc5 Bc4 Bb7 Rfd1 Rc8 Qg4 Bxd5 Rxd5 Qe7 Rcd1 Qe4 Qxe4 Nxe4 Ba6 Nf6 Bxc8 Nxd5 Ba6 Nf6 f4 Re8 Kf2 Kf8 Kf3 Re7 Rd8 Re8 Rxe8 Nxe8 Ke4 Ke7 Bc4 Nc7 Ke5 f6 Kf5 Ne8 Ke4 Nc7 h4 Kd6 Kf5 Ke7 Kg6 Kf8 Kf5 Ke7 Ke4 Kd6 g4 Ke7 b4 Kd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Rc1 dxc4 Bxc4 O-O Nf3 Qxc5 Bb3 Nc6 O-O Qa5 h3 Bf5 Qe2 Ne4 Nd5 e5 Rxc6 exf4 Rc7 Be6 Qe1 Qb5 Ne7 Kh8 Bxe6 fxe6 Qb1 Ng5 Nh4 Nxh3 Kh2 Qh5 Nexg6 hxg6 Qxg6 Qe5 Rf7 Rxf7 Qxf7 Ng5 Ng6 Kh7 Nxe5 Nxf7 Nxf7 Kg6 Nd6 fxe3 Nc4 exf2 Rxf2 b5 Ne3 a5 Kg3 a4 Rc2 Rf8 Kg4 Bd4 Re2 Bxe3 Rxe3 Rf2 b3 Rxg2 Kf3 Rxa2 bxa4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 c5 Qe2 a6 Rfd1 cxd4 Nxd4 Qe7 Ne4 Be5 Nf3 Bb8 Qd2 b5 Be2 Nf6 Nxf6 Qxf6 Qd4 Bb7 Qxf6 gxf6 b3 f5 g3 Bxf3 Bxf3 Ra7 Rc6 Kg7 Be2 Be5 h3 Bf6 Rdd6 Rfa8 Kg2 Be7 Rd2 b4 g4 fxg4 hxg4 a5 f4 Rd8 Rxd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 c6 Bg2 d5 cxd5 cxd5 Nc3 O-O Ne5 e6 O-O Nfd7 f4 f6 Nf3 Nc6 Be3 Nb6 Bf2 f5 Ne5 Bd7 Qd2 Nc8 Qe3 Kh8 Rfd1 Nd6 b3 Rc8 Rac1 Be8 Be1 Bf6 Na4 b6 Nb2 Ne4 Nbd3 g5 Nxc6 Bxc6 Ne5 gxf4 gxf4 Be8 Qh3 Rg8 Kf1 Rxc1 Rxc1 h5 Bb4 a5 Ba3 Bxe5 dxe5 Rg4 Bxe4 dxe4 Bd6 Rxf4 Ke1 Rg4 Qe3 Qg5 Qxg5 Rxg5 Rc8 Rg8 e3 h4 h3 a4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 c4 axb5 axb5 Nd4 Rxa3 bxa3 Nd3 Bxd3 cxd3 Bb2 Qa5 Nf5 Ne5 Bxe5 dxe5 Nb3 Qb6 Qxd3 Ra8 Rc1 g6 Ne3 Bxa3 Ra1 Ra4 Ng4 Bf8 Rc1 Qd6 Nc5 Rc4 Rxc4 bxc4 Nxb7 cxd3 Nxd6 Bxd6 Kf1 Kg7 f3 f5 Nf2 d2 Ke2 Bb4 Nd3 Bc3 Nc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Bg4 Be3 Nfd7 Rd1 Nc6 Be2 Nb6 Qc5 Qd6 e5 Qxc5 dxc5 Nc8 Nb5 Rb8 Nxc7 e6 Nb5 N8e7 Rd2 b6 cxb6 axb6 Bg5 Nf5 b3 h6 Bf6 Bxf3 Bxf3 Nxe5 Bxe5 Bxe5 O-O Rfd8 Rfd1 Rxd2 Rxd2 Rc8 g3 Rc1 Kg2 Kf8 Be4 Ke7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 c4 Nd4 Qf6 N2f3 Nc5 axb5 axb5 Nxb5 Rxa3 Nxa3 Ba6 Re3 Rb8 e5 dxe5 Nxe5 Nbd3 Ng4 Qb6 Rg3 g6 Bxh6 Qxb2 Qf3 Nd7 Bxf8 Kxf8 Kh2 Rb3 Bxd3 cxd3 Qf4 Qxa3 Nh6 Qe7 Rxg6 Qe5 Rg8 Ke7 d6 Ke6 Re8 Kd5 Rxe5 Nxe5 d7 Rb8 Nxf7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Bg4 Be3 Nfd7 Rd1 Nc6 Be2 Nb6 Qc5 Qd6 e5 Qxc5 dxc5 Nc8 h3 Bxf3 Bxf3 Bxe5 Bxc6 bxc6 Bd4 Bf4 O-O a5 Rfe1 a4 Re4 Bh6 Be5 a3 b3 Na7 Rd7 Bc1 Rxc7 Bb2 Na4 Nb5 Rxc6 Rfd8 Rb6 Rd5 Bg3 Nc3 Nxc3 Bxc3 c6 Bd4 Rb7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 e3 h6 Bh4 Bxc3 bxc3 d6 Nd2 g5 Bg3 Qe7 a4 a5 h4 Rg8 hxg5 hxg5 Qb3 Na6 Rb1 Kf8 Qd1 Bc6 Rh2 Kg7 c5 bxc5 Bb5 Nb8 dxc5 d5 Be5 Kf8 Rh6 Ne8 Qh5 f6 Rh7 Ng7 Qf3 Kf7 Qh5 Kf8 Qf3 Kf7 Rh6 Ne8 e4 g4 Qf4 Bxb5 Rxb5 Nd7 Bxc7 Nxc5 Qe3 Nxe4 Nxe4 dxe4 Bxa5 f5 Bb4 Qd7 Qd4 Ra7 Rh7 Ng7 a5 Kg6 Qxd7 Rxd7 Rh4 Rgd8 c4 Rd1 Ke2 Rc1 a6 Rc2 Ke1 Ra2 Rb6 Rd3 c5 Ra1 Ke2 Ra2 Ke1 g3 fxg3 Rxg3 Kf1 Rgxg2 Be1 Rgc2 c6 Ra1 Rh3 f4 Rb4 Kf5 Rb5 e5 Ra5 Rd1 a7 e3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Bf4 Re8 Rad1 Ne4 Nb5 Qf6 Bd3 Nb4 Nc7 Nxd3 Nxe8 Rxe8 Qxd3 Qxb2 Rde1 Qb4 Nd2 Qa4 Qc4 Qxc4 Nxc4 Bc3 Nd2 Bxd2 Bxd2 Bd7 Bf4 Bb5 f3 g5 Bxg5 Bxf1 Kxf1 Nd6 Be7 Nc8 Bxc5 Rd8 Re5 f6 Rf5 b6 Bd4 Ne7 Bxf6 Rxd5 Rg5 Rxg5 Bxg5 Nc6 Ke2 Kf7 Kd3 Ke6 Kc4 Ne5 Kd4 Nc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bg5 Nbd7 Bxf6 Nxf6 Nbd2 Rc8 Nb3 c5 dxc5 Bd5 Rfd1 Bxb3 Qxb3 Qc7 a4 Qxc5 axb5 axb5 Nd4 b4 e3 Rfd8 Rd2 Qb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 O-O O-O d5 Ne5 c6 Bc3 Nfd7 Nxd7 Nxd7 Nd2 Rc8 e4 dxc4 bxc4 b5 Re1 bxc4 Qc2 Qc7 Nf1 e5 Ne3 exd4 Bxd4 Bc5 Bxc5 Nxc5 Nxc4 Rfd8 Rad1 Rxd1 Rxd1 Rd8 Rxd8 Qxd8 h4 Qd4 Qb2 Qxb2 Nxb2 f6 f3 Kf7 Bf1 Bb5 Kf2 Ke6 Bc4 Kd6 Ke3 Nd7 f4 Nb6 Bg8 h6 Nd3 Nd7 Kd4 c5 Kc3 Bc6 Nf2 Nb6 Bb3 Na8 Kd3 Nb6 Bc2 Bb5 Kc3 Na4 Kd2 c4 e5 fxe5 Ne4 Ke6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O dxc4 Bxc4 e5 h3 exd4 exd4 Nb6 Bb3 Bf5 Re1 a5 a3 Re8 Rxe8 Qxe8 Qd2 Nd7 Qf4 Bg6 h4 Qd8 Na4 h5 Re1 b5 Nc3 Qb8 Qe3 b4 Ne4 bxa3 Nxf6 Nxf6 bxa3 Nd5 Bxd5 cxd5 Ne5 Qd8 Qf3 Ra6 Rc1 Kh7 Qh3 Rb6 Rc8 Qd6 Qg3 a4 Ra8 Qe6 Rxa4 Qf5 Ra7 Rb1 Kh2 Rc1 Rb7 Rc2 f3 Rd2 Nd7 Rxd4 Nf8 Kh6 Rb4 Rc4 Rxc4 dxc4 Qd6 c3 Qd4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 d4 cxd4 Qxd4 Bg7 Nc3 d6 Rd1 Nbd7 b3 Rc8 Bb2 O-O Qe3 Re8 Rac1 a6 Ba1 Rc5 a4 Qa8 Ne1 Rf5 Bxb7 Qxb7 f3 h5 Ng2 Rc5 Bb2 Rcc8 Ba3 Nc5 Rb1 Ne6 Qd3 Nc7 Nf4 b5 cxb5 axb5 Nxb5 Nxb5 Qxb5 Qxb5 axb5 Rb8 Bb2 Rb7 b6 Reb8 b4 Nd7 Bxg7 Kxg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Bb4 Bd2 a5 O-O O-O Bg5 Be7 Qc2 h6 Bxf6 Bxf6 Nc3 g6 Rad1 d6 h4 h5 e4 Nd7 e5 Bg7 d5 Nxe5 Nxe5 Bxe5 dxe6 Bxg2 exf7 Kxf7 Kxg2 Bxc3 Qxc3 Qf6 Qxf6 Kxf6 a4 Rae8 Rfe1 Rxe1 Rxe1 Rd8 Rd1 c6 Kf3 Ke5 Ke3 Rf8 f3 Rh8 Re1 Rb8 Kd3 Kf6 Re4 d5 cxd5 cxd5 Re2 b5 Kd4 bxa4 Kxd5 Rb3 Ke4 Rb4 Kd5 Rb5 Kd4 Rb4 Kd5 Rb3 Ke4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 e6 d4 cxd4 Nxd4 Nf6 Nxc6 bxc6 e5 Nd5 Ne4 Qc7 f4 Qb6 c4 Bb4 Ke2 f5 Nf2 Ba6 Kf3 Ne7 Be3 Bc5 Bxc5 Qxc5 Rc1 g5 Qd2 Ng6 g3 Rg8 b4 Qe7 Qd6 Qxd6 exd6 c5 Ke3 cxb4 Nd3 Rc8 Nxb4 Bb7 Rg1 Rc5 Nd3 Ra5 Rc2 Ra3 Rb2 Be4 Kd4 Kf7 Nc5 Bc6 Bg2 Rc8 Nd3 gxf4 gxf4 Ra4 Rc1 Bxg2 Rxg2 Kf6 Rxg6 hxg6 Ne5 g5 Nxd7 Kg7 Ne5 gxf4 d7 Rd8 Kc5 Kf6 Kd6 Rxa2 Kc7 Rh8 Nf3 e5 c5 e4 Nd4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 dxc3 Qxd1 Kxd1 Nd7 Bf4 c6 Bd3 f6 e5 Nc5 Bc2 Bg4 Ke2 Bg7 Be3 Nd7 exf6 exf6 h3 Be6 Bb3 Bxb3 axb3 a6 Rhe1 Kf7 Rad1 Rhd8 Rd3 Nf8 Red1 Rxd3 Rxd3 Re8 Kd2 Ne6 Rd7 Re7 Rxe7 Kxe7 b4 f5 Kd3 Kd6 Nd2 b5 Nb3 Kd5 Nc5 Nxc5 Bxc5 a5 b3 axb4 Bxb4 h5 c4 Ke6 Bd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 d6 g3 f5 Bg2 Nf6 e3 g6 Nge2 Bg7 O-O O-O d3 c6 b4 Be6 b5 d5 bxc6 bxc6 Qa4 g5 Ba3 Rf7 Rac1 Rd7 Bb4 a5 Ba3 Bf7 cxd5 cxd5 Nb5 Na6 Rc6 Nb4 Bxb4 axb4 Qxb4 Rxa2 Nec3 Ra8 Nd6 Bf8 Nxf7 Rxf7 Qb3 Ra3 Qc2 e4 dxe4 dxe4 Rd1 Qe8 Nb5 Ra2 Qc4 Ra4 Qc2 Ra2 Qxa2 Qxc6 Nd4 Qd5 Qc2 Ng4 h3 Ne5 Nxf5 Nd3 g4 h5 Qc8 Qa2 Rf1 Kh7 Bxe4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nc6 Nc3 a6 Bg5 Rb8 Rc1 Bg4 d5 Bxf3 exf3 Ne5 Qe2 Re8 Rfe1 c5 dxc6 Nxc6 Rcd1 h6 Bc1 Nd7 f4 Nd4 Qd2 Qa5 Re4 Nf5 Qe1 Nc5 Nd5 Qxa2 Re2 e6 b4 Qxc4 bxc5 Qxc5 Ne3 Nd4 Red2 b5 Bb2 e5 fxe5 dxe5 Nd5 Rbd8 Nc3 Kh7 h4 b4 Ne4 Qb5 h5 f5 hxg6 Kxg6 Rd3 fxe4 Qxe4 Kf7 Qg4 Rd6 Qh5 Kf8 Be4 Rf6 Kg2 a5 Bxd4 exd4 Qxb5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Qb3 Qb6 Nc3 Bg4 c5 Qc7 Bf4 Qc8 Ne5 Nbd7 f3 Bh5 g4 Bg6 h4 h6 Nxg6 fxg6 e3 e5 dxe5 Ng8 Qc2 Ne7 Ne2 Qc7 b4 O-O-O Nd4 Nxe5 Qh2 Nxf3 Nxf3 Qd7 Nd4 g5 hxg5 Ng6 gxh6 gxh6 Bh3 Nxf4 Qxf4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Nbd2 c5 a3 Bxd2 Bxd2 cxd4 Nxd4 Nc6 Bc3 O-O Rc1 Ne4 e3 Qb6 Bd3 Nxc3 Rxc3 Nxd4 exd4 d6 O-O h6 b4 e5 c5 dxc5 dxc5 Qc7 Qe2 Be6 Re1 f6 Qe4 Rfd8 c6 b6 Bc4 Bxc4 Qxc4 Kf8 h3 Rd4 Qe6 Re8 Qf5 Kg8 b5 Red8 Qe6 Kh8 Rg3 R4d5 a4 Rd4 Qb3 R8d5 Qa3 Rc5 Rd3 Rcc4 Rxd4 Rxd4 Qf8 Kh7 Rc1 Rd8 Qb4 Rd4 Qb3 e4 Qg3 Qxg3 fxg3 Rd8 g4 Kg6 Kf2 f5 Ke3 Kf6 gxf5 Kxf5 Rf1 Ke5 Rc1 Rd3 Ke2 Rd8 c7 Rc8 Ke3 Kd5 Rc6 Ke5 h4 Kd5 g4 Ke5 h5 Kd5 Rc1 Ke5 Rc4 Kd5 Rc1 Ke5 Rc2 Kd5 Rd2 Ke5 Rd7 Kf6 Kxe4 g6 Kd5 gxh5 gxh5 Kg5 Kc6 Kxh5 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 g3 c5 Nf3 cxd4 Bg2 e5 O-O d6 c3 dxc3 Nxc3 Nf6 Bg5 Be7 Nd2 Nbd7 Nc4 Qc7 b3 h6 Nb5 Qc5 Nbxd6 Kf8 Be3 Qc7 Rc1 Ng4 Qd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 cxd4 exd4 d5 c5 Ne4 Bd2 Nxd2 Qxd2 b6 a3 Bxc3 Nxc3 bxc5 dxc5 a5 Rc1 Nc6 Bb5 Bd7 O-O O-O Na4 Ne5 Bxd7 Nxd7 Qd4 Qc7 c6 Nf6 Rc2 Rab8 b4 axb4 axb4 Ne4 Rb1 Nd6 Rc5 Qa7 Nc3 Nf5 Qf4 d4 Ne4 d3 c7 Rbc8 b5 Qa2 Re1 Qb2 h4 Nd4 Kh2 Nxb5 Re3 d2 Rd3 Qb4 Nf6 gxf6 Qxb4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 c4 e6 b3 Nf6 Bb2 Be7 g3 O-O Bg2 c5 O-O Nc6 e3 dxc4 bxc4 b6 Qe2 Ba6 Nc3 Rc8 Rfd1 Qd7 d3 Rfd8 Ne1 Bb7 f4 a6 g4 Ne8 g5 Na5 a4 Bxg2 Qxg2 Nc6 Ne4 Bf8 Nf3 Rb8 h4 b5 axb5 axb5 Ne5 Nxe5 Bxe5 Rbc8 Rd2 Qc6 Rad1 bxc4 dxc4 Rxd2 Rxd2 Qa6 Qe2 Qa3 h5 Qa5 g6 f6 Bc3 Rd8 Bxa5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Nbd2 d5 e3 O-O a3 Be7 Bd3 b6 O-O Bb7 b4 c5 bxc5 bxc5 Rb1 Qc7 cxd5 exd5 dxc5 Nbd7 Bb2 Nxc5 Qc2 Rac8 Bf5 Ne6 Qxc7 Rxc7 Be5 Rd7 Nd4 Nxd4 Bxd4 Bc8 Bxd7 Bxd7 Rb7 Rd8 Rxa7 Ne8 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 g6 e4 Bg7 c3 d6 Bg5 Nf6 Nd2 h6 Bh4 c6 f4 Qc7 Bd3 e5 fxe5 dxe5 Ngf3 exd4 cxd4 O-O O-O Be6 Qe1 Nbd7 e5 Nd5 Ne4 Qb6 Qf2 Nf4 Bb1 Bc4 Re1 Nd3 Bxd3 Bxd3 Nd6 g5 Bg3 Bg6 h4 g4 Nh2 h5 Nf1 c5 Ne3 cxd4 Nd5 Qc5 Ne7 Kh7 Rac1 Qa5 Qxd4 Qxa2 Ne4 Rad8 Ng5 Kh8 Rcd1 Bc2 Rd2 Bg6 Qd6 Bh7 Red1 Qa5 Kh1 Bg8 Qg6 Bxe5 Qh6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 Nc6 e3 e6 d4 d5 a3 cxd4 exd4 Be7 Bd3 O-O O-O dxc4 Bxc4 b6 Qd3 Bb7 Ba2 Qd6 Bg5 Rfd8 Rad1 Nh5 Bc1 Nf4 Qe4 Ng6 Rfe1 Bf8 Qe2 Rac8 d5 exd5 Nxd5 Qb8 Ng5 Re8 Qh5 Rxe1 Rxe1 h6 Qxg6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 g3 a6 Bg2 b5 b3 Bb7 Nc3 b4 Na4 d6 O-O Nbd7 d4 Be7 Nb2 O-O Nd3 a5 Bb2 c5 e3 Qb6 a4 Rfd8 Qe2 Be4 Rfd1 Qb7 Nfe1 Bxg2 Nxg2 Rac8 Ngf4 Nf8 d5 exd5 Qf3 Rd7 Nxd5 Nxd5 cxd5 c4 bxc4 Rxc4 Rac1 Rdc7 e4 Rxc1 Rxc1 Rxc1 Nxc1 Nd7 Nb3 Qc7 Qd3 Bf6 Bxf6 Nxf6 e5 dxe5 d6 Qc6 Nxa5 Qxa4 Nc4 h5 Nb6 Qa1 Kg2 Qd4 Qxd4 exd4 Kf3 b3 Nc4 Kf8 Ke2 Ke8 Kd3 Kd7 Kxd4 Ke6 Kc3 Ne4 Kxb3 Nxf2 Ne5 Kxd6 Nxf7 Kd5 Kc2 Ke4 Ng5 Ke3 Ne6 g6 Nf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 Qc2 Nxc3 bxc3 Be7 e3 O-O Bd3 h6 e4 c5 O-O Ba6 Bf4 cxd4 Bxa6 Nxa6 cxd4 Qc8 Qe2 Qb7 Rfd1 Rfe8 Ne5 Rad8 Rd3 Bf6 Rad1 Nb8 Ng4 Bg5 Bxg5 hxg5 e5 Qe7 Qe4 Nd7 f4 Nf8 fxg5 Qxg5 Rg3 Nh7 Rf1 g6 Qf3 Kh8 h4 Qf5 Qe3 Qh5 Rxf7 Rf8 Rxf8 Rxf8 Nf2 Qxh4 Rh3 Qf4 Ne4 Qxe3 Rxe3 Rd8 Ng5 Nxg5 Rc3 Rxd4 Rc8 Kg7 Rc7 Nf7 Rxa7 Rd5 Rb7 b5 Kf2 Rxe5 Kf3 Kf6 Kf4 g5 Kf3 Nd6 Rb6 Nc4 Rb8 Re3 Kf2 Kf5 Rxb5 e5 Rc5 Rc3 a4 Kf4 g3 Ke4 a5 Rc2 Kg1 Kf3 a6 Kxg3 Kf1 Ne3 Ke1 Rxc5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 Nf6 e5 Nd5 Bxc4 Nb6 Bd3 Nc6 Be3 Be6 Ne2 Nb4 Be4 Bd5 f3 e6 Nbc3 Qd7 O-O Nc4 Bc1 Rd8 b3 Nb6 Bg5 Be7 Bxe7 Qxe7 Qc1 O-O Nb5 Qd7 Na3 f5 exf6 Rxf6 Qc3 Nc6 Nc2 Qf7 Ne3 Qd7 Rad1 Bxe4 fxe4 Rxf1 Nxf1 Nc8 d5 exd5 exd5 N6e7 Nf4 Nd6 Ng3 Rf8 Ngh5 Rf7 Qxg7 Rxg7 Nf6 Kh8 Nxd7 Nxd5 Nxd5 Rxd7 Nxc7 Rxc7 Rxd6 Rc1 Kf2 Rc2 Kf3 Rxa2 Rd8 Kg7 Rd7 Kg6 Rxb7 a5 h4 a4 b4 Rb2 g4 h5 Rb6 Kg7 g5 a3 Ra6 Rxb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 c5 Bg2 Qa5 Qd2 Nc6 d5 Ne4 Qxa5 Nxa5 Na3 Nd6 Nd2 Rb8 e3 a6 Rb1 b5 Bf1 e6 dxe6 dxe6 Be2 O-O O-O Nc6 Nb3 Nxc4 Nxc4 bxc4 Bxc4 Ne5 Be2 c4 Nd4 Bb7 f3 Nd3 Bd2 Bxd4 exd4 Bd5 Bc3 Nb4 a3 Nc6 Kf2 Na7 Rfc1 Rfc8 Rc2 Nb5 Ke3 Nd6 Kd2 h5 Rbc1 Nf5 g4 hxg4 fxg4 Nd6 Ke3 Rb3 Bd1 Kg7 Rf2 Rh8 Bxb3 cxb3 Ke2 Ne4 Rcf1 Bc4 Ke3 Bxf1 Kxe4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 d6 g3 f5 Bg2 Nf6 e3 g6 Nge2 Bg7 O-O O-O d3 c6 b4 Be6 b5 d5 bxc6 bxc6 cxd5 cxd5 d4 e4 Ba3 Re8 Na4 g5 Nc5 Bf7 Qd2 Nc6 Rab1 Na5 Nb7 Nc4 Qc1 Qd7 Nc5 Qc6 Bb4 Bh5 Nc3 Bf3 Re1 Kh8 Bf1 Ng4 h3 a5 Nxd5 Ncxe3 Nxe3 Nxe3 Qxe3 f4 gxf4 gxf4 Qxf4 axb4 Bg2 Qg6 Qg3 Qxg3 fxg3 Bxd4 Kh1 Bxg2 Kxg2 Rxa2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 cxd5 Nxd5 Bxe7 Qxe7 Nxd5 exd5 Rc1 Be6 Qa4 c5 Qa3 Rc8 Be2 Kf8 dxc5 bxc5 O-O a5 Rc3 Nd7 Rfc1 Rcb8 Rb3 c4 Rxb8 Rxb8 Qxa5 Rxb2 Nd4 Kg8 Ra1 Nc5 Qa8 Kh7 Qa3 Rb6 Bd1 g6 Bc2 Bd7 h3 Qd6 Qa5 Ba4 Bxa4 Ra6 Qb5 Rxa4 a3 c3 Nc2 Qc7 Rb1 Ra7 Qe8 Rb7 Rb4 Kg7 g3 Ne6 Qa4 Ng5 h4 Ne4 Kg2 Ra7 Qb5 Qe5 Qb6 Qf5 f3 Ng5 hxg5 Qxc2 Kg1 Qd1 Kg2 Qe2 Kh3 Qxf3 Qxa7 Qh1 Kg4 h5 Kf4 Qf1 Ke5 Qf5 Kd6 Qe6 Kc7 Qe7 Kb6 Qxa7 Kxa7 c2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c3 Bg7 Bg5 d5 Nbd2 Nbd7 b4 b6 e3 O-O b5 Qe8 Be2 e5 O-O Bb7 a4 Ne4 Rc1 a6 Bh4 axb5 axb5 Ra7 Bg3 Nxg3 hxg3 Qe7 Nxe5 Nxe5 dxe5 Bxe5 Nf3 Bd6 Ra1 Rfa8 Rxa7 Rxa7 Qc2 Ra5 Nd4 h5 Bf3 Qe5 Be2 Ra3 Rc1 Ra8 Rd1 Ra3 Rc1 Ra8 Rd1 Ra3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bc4 e6 O-O Nf6 d3 Be7 Re1 O-O Bb3 Nc6 c3 b5 a4 b4 Nbd2 Ba6 Bc4 Bb7 d4 cxd4 cxd4 d5 exd5 Nxd5 Ne4 Rc8 Bd3 b3 Qxb3 Ndb4 Bb1 Nxd4 Nxd4 Qxd4 Be3 Qd5 Qxd5 exd5 Nc3 Bf6 Bf5 d4 Bxc8 Rxc8 Nb5 dxe3 Nd6 exf2 Kxf2 Rc2 Kg3 Bc6 Rad1 Rxg2 Kh3 Rxb2 Ne4 Rb3 Kg4 Bxe4 Rxe4 g6 Rd7 h5 Kf4 g5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 b3 e5 Bb5 Bd7 Bxd7 Qxd7 O-O Nc6 c4 g6 d3 Bg7 Nc3 f5 Nd5 Nge7 Bg5 O-O Bxe7 Nxe7 Nxe7 Qxe7 Nd2 Qh4 Qe2 Rf6 Nb1 g5 g3 Qh3 f3 g4 Nc3 f4 Nd5 Rh6 Qg2 fxg3 hxg3 Qh5 Ne7 Kh8 Nf5 Rg6 fxg4 Qxg4 Rf3 Bh6 Raf1 Bg5 Kf2 a5 a4 Ra6 Ke2 Rb6 Rb1 Rb4 Kd1 d5 Qe2 b5 axb5 a4 cxd5 axb3 Rf2 Qh3 Rh2 Qxh2 Qxh2 Rxb5 Qb2 Bf6 Qa3 Rg8 Rxb3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 Ne2 cxd4 exd4 d5 O-O dxc4 Bxc4 O-O a3 Bd6 Qd3 e5 d5 e4 Nxe4 Ne5 Qd4 Nxe4 Qxe4 Bg4 f4 Nxc4 Qxc4 Rc8 Qd3 Re8 Nc3 Qf6 Bd2 h5 Rae1 Bf5 Qf3 Bc5 Kh1 Qd4 Qd1 b5 Rxe8 Rxe8 Re1 Rxe1 Qxe1 Bd3 h3 a5 Kh2 Kf8 Nd1 Bd6 Kh1 Qxd5 Bxa5 Bxf4 Bd2 Bc7 Bb4 Kg8 Qe8 Kh7 Nf2 Bg6 Qe3 Qc4 Kg1 Bf4 Qe1 f6 Bc3 Bb8 Qe3 Bd6 Qd4 Qc6 Kf1 Bc5 Qd2 Bb6 Qe2 Qd5 Qd2 Qc4 Qe2 Qf4 Bd2 Qf5 Bc3 Bf7 Qf3 Qb1 Nd1 Bc7 Ke1 h4 Kd2 Bg6 Bd4 Qe4 Qxe4 Bxe4 Nc3 Bxg2 Nxb5 Bh2 Bf2 g5 Nd4 Kg6 b4 Bxh3 b5 Bg2 a4 h3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c3 Bg7 Bg5 d5 Nbd2 Nbd7 b4 Ne4 Nxe4 dxe4 Nd2 h6 Bh4 e3 fxe3 g5 Bf2 Nf6 g3 a5 a3 h5 Bg2 h4 Ne4 h3 Nxf6 exf6 Be4 f5 Bd3 Qd5 Kd2 Be6 Rg1 O-O b5 c5 c4 Qd6 d5 Bd7 Rb1 Rae8 Qc2 Re5 Rgf1 Rfe8 Bg1 Qg6 Rf3 Re4 Rbf1 R8e7 R1f2 g4 Rf4 Bh8 Bxe4 fxe4 Rf1 f5 R1f2 Qh7 Rf1 Qh5 Bf2 Qe8 Rb1 Rf7 Rf1 Qe5 Rb1 Rf6 Be1 Rd6 Bf2 Qe7 Ke1 Be5 Kf1 Bxf4 gxf4 Qf6 Qb2 Qxb2 Rxb2 a4 Bg3 Kf8 Ke1 Ke8 Kd2 Rg6 Kc3 Kd8 Rb1 Kc7 Rb2 Kd6 Rb1 Be8 Rb2 Bf7 Rb1 Rg8 Rd1 Ra8 Rd2 Ra5 Rd1 Be8 Rd2 Bxb5 cxb5 Rxb5 Rb2 Rxb2 Kxb2 Kxd5 Kc3 b5 Bh4 Kc6 Be7 Kb6 Kb2 b4 axb4 cxb4 Bxb4 g3 hxg3 h2 g4 h1=Q gxf5 Qd1 Kc3 Qb3 Kd4 Qxb4 Ke5 Qe7 Kd5 Kb5 f6 Qxf6 Kxe4 Kc4 Kf3 a3 e4 a2 e5 Qc6 Kg4 a1=Q Kf5 Qg1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 O-O Nc6 c4 Nf6 Nc3 e6 d4 cxd4 Nxd4 Be7 b3 O-O Be3 a6 Qe2 Rfe8 Rfd1 Nxd4 Bxd4 Qc6 f3 Nd7 Rac1 Rac8 Be3 Bf6 Qd2 Be7 a4 b6 Rb1 Red8 Ne2 Qb7 Nd4 Ne5 Bf4 Ng6 Bg3 h6 Qe2 Bf6 Nc2 d5 exd5 exd5 Ne3 dxc4 Nxc4 Bd4 Kh1 Qc6 Rd2 Bc5 Rxd8 Rxd8 Rd1 Rxd1 Qxd1 b5 axb5 axb5 Ne5 Nxe5 Bxe5 Qe6 Bg3 Bf8 h3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bg5 Ne4 Bh4 Nxc3 bxc3 dxc4 Qa4 Qd7 Qxc4 b6 Bg3 c5 Ne5 Bxe5 Bxe5 Ba6 Qb3 f6 Bxb8 Rxb8 e3 Bxf1 Kxf1 cxd4 cxd4 Rc8 f3 Qc6 Kf2 Qc2 Kg3 Qxb3 axb3 a5 Rhc1 Kd7 Kf2 Rxc1 Rxc1 Rc8 Rxc8 Kxc8 Ke2 b5 Kd3 Kc7 e4 e6 g4 Kd6 h4 h5 gxh5 gxh5 Kc3 Kc6 Kd3 a4 b4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 f4 O-O Kh1 Qc7 a4 Nc6 Be3 Re8 g4 Nxd4 Qxd4 Nd7 g5 b6 Bf3 Bb7 Bg2 Bf8 Qd2 Rac8 f5 Qc4 g6 hxg6 fxg6 fxg6 Qf2 Nf6 Qg3 e5 Qxg6 Qe6 Rae1 b5 axb5 axb5 Bg5 Nh7 Qxe6 Rxe6 Be3 Ree8 Rf2 Nf6 Bg5 b4 Nd5 Nxd5 exd5 Rc5 Rd2 Rc4 Re4 Rec8 Rxc4 Rxc4 Bh3 Bc8 Be6 Bxe6 dxe6 Rc7 Kg2 Kh7 Rf2 Be7 Be3 Kg6 Kg3 Bf6 h4 d5 Kg4 Rc4 Kh3 Rc6 Kg4 Rxe6 Bg5 Bxg5 hxg5 e4 Rd2 e3 Re2 Re4 Kf3 Kxg5 Rg2 Kf6 Ke2 g5 Rh2 g4 Rh8 Kg5 Rg8 Kh4 Rd8 g3 Kf3 Kh3 Rh8 Rh4 Rg8 Kh2 Rxg3 Re4 Rg2 Kh3 Re2 Rc4 Kxe3 Kg4 Kd3 Kf4 Re1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 a6 Be2 b5 Qb3 c5 dxc5 Nbd7 e5 Ng4 O-O Ngxe5 Nxe5 Nxe5 Rd1 Qc7 Nd5 Qb7 Be3 Be6 f4 Rad8 fxe5 Bxd5 Rxd5 Qxd5 Qxd5 Rxd5 Bf3 Rxe5 Bd4 Re1 Rxe1 Bxd4 Kf1 Bxc5 Rd1 Bd6 Rc1 Bxh2 Rc6 Bd6 Rxa6 b4 a4 bxa3 bxa3 Rb8 a4 Rb2 a5 Ra2 Ra8 Kg7 a6 h5 Bd5 Ra5 Bc6 h4 Ke2 g5 Kf3 f5 Bd7 g4 Ke2 Kf6 Rf8 Kg6 Bxf5 Kg7 Rc8 h3 gxh3 g3 Be4 Rxa6 Bd5 Ra5 Bc6 Ra2 Kf1 Rf2 Kg1 Rc2 Bd7 Bc5 Rxc5 Rxc5 Kg2 Rc3 Bg4 Kf6 Bf3 e5 h4 Kf5 Bh5 Kf4 Bf7 Rc2 Kf1 e4 Bh5 e3 Be2 g2 Kxg2 Rxe2 Kf1 Rc2 Ke1 Rc1
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 g6 Nf3 Bg7 Qb3 O-O Bd2 e6 Bd3 Nbd7 O-O a5 Rfe1 Re8 e4 a4 Qc2 dxe4 Nxe4 Nxe4 Bxe4 c5 d5 Nf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 e6 g3 f5 Bg2 Nf6 Nf3 Be7 c4 d5 O-O O-O Nbd2 c6 Ne5 Nbd7 Nd3 Ne4 Qc2 Bf6 Nf3 Kh8 b3 Qe8 Ba3 Rg8 Rac1 a5 Bb2 Qh5 a4 Rd8 Ba3 Rg8 Rb1 Ng5 Rfe1 Qf7 Nxg5 Bxg5 Bd6 Bf6 e3 g5 b4 axb4 Rxb4 Nf8 Reb1 Ng6 Nc5 Ra7 Qb3 Rg7 Qc2 Qg8 h3 Qd8 Bb8 Ra8 cxd5 cxd5 Be5 Nxe5 dxe5 Bxe5 Nxb7 Bxb7 Rxb7 Rxb7 Rxb7 Rb8 Qc6 Rxb7 Qxb7 Qc7 Qb5 Kg7 a5 d4 a6 dxe3 a7 exf2 Kh1 Qxa7 Qxe5 Kf7 Bf1 h5 Bc4 f1=Q Bxf1 Qa8 Kh2 Qd8 Be2 Qd2 Kg1 Qe1 Kg2 Qd2 Kf2 g4 h4 Qd8 Ke3 Qf6 Qa5 e5 Bc4 Kg6 Qc7 Qg7 Qd6 Qf6 Qd7 Qb6 Kd3 Qb1 Kc3 Qe1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Nbd2 c5 e3 O-O a3 Bxd2 Qxd2 cxd4 exd4 b6 Be2 d5 b3 Ba6 Qb2 dxc4 bxc4 Nc6 Bg5 Rc8 Rd1 Na5 c5 Bxe2 Qxe2 Qd5 Bxf6 gxf6 cxb6 axb6 O-O Rc3 Qd2 Rxa3 Qf4 Kg7 Qg4 Kh8 Qf4 Kg7 Qg4 Kh8 Qf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 Bb4 a3 Ba5 O-O O-O Qc2 dxc4 Bxc4 Bc7 Ba2 e5 h3 h6 e4 Re8 Be3 Nh5 Rad1 exd4 Bxd4 Qe7 e5 Nf8 Nb5 Ne6 Bxe6 fxe6 Nxc7 Qxc7 Qg6 Qf7 Qxf7 Kxf7 Be3 Rf8 Rd4 Kg8 Rfd1 b6 Rh4 Rf5 Nd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 cxd5 exd5 Ne2 c5 O-O Nc6 a3 cxd4 exd4 Bxc3 bxc3 Ne7 Qc2 Bd7 Bg5 Ng6 f4 h6 Bxf6 Qxf6 f5 Ne7 Ng3 Nc8 Rf4 Nd6 Qf2 Rfe8 Nh5 Qd8 Nxg7 Ne4 Bxe4 Rxe4 f6 Kh7 Rxe4 dxe4 Qf4 Bc6 Re1 Qf8 c4 Qxa3 Nf5 Qf8 Re3 Bd7 Rg3 Bxf5 Qxf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c5 Nf3 cxd4 Nxd4 Qc7 b3 Bg7 Bg2 d5 cxd5 Nxd5 O-O Qd7 Bb2 O-O Qc1 Rd8 Rd1 Nc6 Nxc6 Qxc6 Qxc6 bxc6 Bxg7 Kxg7 Rc1 Bg4 Kf1 a5 h3 Be6 Nc3 Nxc3 Rxc3 Rd2 Rxc6 Rad8 Rcc1 Rb2 Rcb1 Rdd2 Rxb2 Rxb2 Ke1 Bf5 Kd1 g5 Bd5 Kf6 Bc4 e6 g4 Bg6 a4 Ke5 Rc1 Kd4 Bb5 Rxb3 Rc4 Ke5 Rc7 Rxh3 Ra7 Kd4 Kd2 Rh2 f3 h5 Rxa5 hxg4 fxg4 Rg2 Bc6 Rxg4 Rb5 Kc4 Rb7 Rg1 Be8 Ra1 Bxf7 Bxf7 Rxf7 Rxa4 Rg7 Ra5 e3 Ra2 Ke1 e5 Rxg5 e4 Rg8 Kd3 Rb8 Re2 Kf1 Rc2 Ke1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 Qa4 Bd7 Qh4 Bc6 g3 Bg7 Bg2 Nd7 Nd4 Bxd4 Nxd5 Bg7 O-O e6 Qxd8 Kxd8 Nc3 Bxg2 Kxg2 Ke7 d3 Rhc8 Bg5 Kf8 Rac1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 e6 g3 d5 cxd5 exd5 d4 Nc6 Bg2 Be7 O-O O-O Be3 c4 Ne5 h6 Qa4 a6 Nxc6 bxc6 b3 Rb8 bxc4 Rb4 Qd1 Rxc4 Na4 Bf5 Nb2 Rc3 Rc1 Rxc1 Qxc1 Qa5 Qxc6 Qxa2 Nd3 Be4 Bf3 a5 Nc5 Bxf3 exf3 Re8 Nd7 Rd8 Nxf6 Bxf6 Qb6 a4 Bxh6 a3 Bc1 Rc8 h4 Rc3 Be3 Qc2 Qa5 a2 h5 Be7 Qa8 Bf8 h6 Ra3 Qxd5 a1=Q hxg7 Bxg7 Qd8 Bf8 Rxa1 Rxa1 Kg2 Qh7 Qg5 Qg6 Qh4 Rd1 g4 f6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 d6 Nc3 Nbd7 e4 e5 Nf3 c6 Be2 Be7 O-O O-O d5 a6 Be3 cxd5 cxd5 b5 Nd2 Nxe4 Ncxe4 f5 a4 bxa4 Qxa4 fxe4 Nxe4 Nf6 Nxf6 Rxf6 b4 Rf8 Qc2 Bg5 Bd3 h6 b5 Bb7 bxa6 Bxe3 fxe3 Rxf1 Bxf1 Bxd5 Rd1 Bf7 Qe4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Qa5 Bd2 Qd8 Bg5 Qa5 Bd2 Qd8 Nxf6 Qxf6 Bd3 Qg6 O-O Be7 c3 d5 Qe2 Bg4 f3 dxe4 Bxe4 Bf5 Nc2 O-O a4 Bxe4 fxe4 Rad8 Be1 b4 Bf2 b3 Ne1 Na5 Nd3 Qe6 Nb4 Nc4 Nd5 Rxd5 exd5 Qxd5 Rfd1 Qc6 Rd3 f5 Rad1 Qe4 Re1 Qxe2 Rxe2 Rd8 Rxd8 Bxd8 Re1 Be7 Rd1 Bd6 Kf1 Kf7 Ke2 Ke6 Be3 f4 Bc1 g5 g4 Bc5 Rd8 e4 Re8 Kd5 Rd8 Nd6 Rd7 f3 Ke1 e3 c4 Ke6 Bxe3 Bb4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 Nc6 e3 e6 d4 d5 a3 a6 dxc5 Bxc5 b4 Ba7 Bb2 O-O Qc2 Qe7 Rd1 Rd8 Be2 d4 exd4 Nxd4 Nxd4 Bxd4 O-O e5 Na4 Bd7 Bxd4 Bxa4 Qxa4 Rxd4 Qc2 Rad8 Rxd4 exd4 Bd3 g6 Qd2 Ng4 h3 Ne5 Re1 Qf6 c5 Kg7 Re4 Qf5 Qe2 Nxd3 Qxd3 h5 h4 Rd7 f3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 a6 Qb3 c5 dxc5 Nbd7 Qb4 Qc7 Na4 a5 Qc4 Ne5 Nxe5 Qxe5 Bd3 Rd8 f3 Be6 Qc2 Nd5 a3 Bd7 Bc4 Bxa4 Qxa4 Nc3 bxc3 Qxc3 Ke2 Qxa1 Qb3 e6 Rf1 Qe5
Result: 0-1
Processing game...
FEN: 
Moves: d4 g6 Nf3 Bg7 g3 c5 d5 d6 c4 b5 cxb5 Nf6 Nc3 a6 bxa6 Qa5 Bd2 O-O Bg2 Bxa6 O-O Nbd7 Qc2 Rfb8 Rfd1 Ng4 Be1 Nge5 Nxe5 Bxe5 Rdb1 Ra7 a4 Rab7 b3 Qc7 Ra2 Qc8 Nd1 Bg7 f4 c4 b4 Rxb4 Rxb4 Qc5 e3 Rxb4 Bxb4 Qxb4 Bf1 Nc5 Nf2 Qb3 Be2 h5 a5 h4 Kg2 Bc3 Bd1 Qb4 Ne4 Bg7 Nxc5 Qxc5 e4 g5 fxg5 c3 Be2 Bc8 gxh4 Qe3 Ra1 Bh3 Kh1 Qf2 Rg1 Bd4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 d5 g3 Be7 Bg2 O-O O-O c5 cxd5 exd5 dxc5 Bxc5 Qc2 Bb6 Nc3 Nc6 Bg5 Be6 Rad1 h6 Bh4 Qe7 Qa4 Rfe8 Rxd5 Bxd5 Nxd5 Qe4 Nxf6 gxf6 Qb5 Qg6 e3 Rad8 a4 Re4 Ra1 Rb4 Qe2 a5 h3 Rd7 g4 f5 Bg3 fxg4 Nh4 Qg5 hxg4 Rxg4 Nf3 Qh5 Qb5 Qc5 Qxc5 Bxc5 Bh3 Rxg3 fxg3 Bxe3 Kf1 Rd3 Bc8 Rb3 Ra3 Rxa3 bxa3 b6 Bd7 Ne7 Nh4 Bc1 Nf5 Ng6 Be8 Ne5 Ke2 h5 Bb5 Kh7 Nd6 Bxa3 Nc4 Nxc4 Bxc4 Kg6 Kf3 f6 Bb5 Kg5 Be8 Bd6 Bb5 h4 gxh4 Kxh4 Be8 f5 Bb5 Kg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 d6 g3 f5 Bg2 Nf6 d3 g6 e4 Bg7 Nge2 O-O O-O c6 h3 Be6 exf5 gxf5 Be3 Qe7 Qd2 Kh8 Rae1 Nbd7 f4 Rg8 Kh2 Qf7 Bf3 Rae8 b4 e4 dxe4 fxe4 Bxe4 Bxh3 Kxh3 Nxe4 Qd3 Bxc3 Qxc3 Nxc3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 c4 e6 d4 Nf6 Nc3 Bb4 cxd5 exd5 Bg5 Nbd7 Rc1 c5 a3 Bxc3 Rxc3 cxd4 Qxd4 O-O e3 h6 Bh4 g5 Bg3 Ne4 Rc1 Qa5 Nd2 Re8 Be2 Ndf6 f3 Nxd2 Qxd2 Qb6 Kf2 Bf5 Qd4 Qxd4 exd4 Re6 Rc7 Rb6 b4 Nh5 Rhc1 Nxg3 hxg3 a5 b5 Re8 g4 Rbe6 Bf1 Bg6 Rxb7 Re3 Rb8 Rxf3 gxf3 Rxb8 Rc5 Rd8 b6 Rb8 Rc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 Nf3 f5 g3 Nf6 Bg2 d5 c4 c6 O-O Bd6 b3 Qe7 a4 O-O Ba3 Bxa3 Nxa3 Nbd7 a5 b6 Qd2 Ne4 Qb2 Bb7 b4 bxa5 bxa5 Rab8 Rfb1 c5 Qc1 Rfc8 Qe3 cxd4 Qxd4 Qc5 Rb5 Qxd4 Nxd4 Kf7 cxd5 Bxd5 Nxf5 Ndf6 Bxe4 Nxe4 f3 exf5 Rxd5 Nc3 Rxf5 Ke6 e4 g6 Rg5 Rb2 Kh1 Kf6 h4 Rf2 Rc1 Rd8 Nc4 Ne2 Rb1 Rxf3 Kg2 Rdd3 Ne5 Rxg3 Kf2 Rc3 Ng4 Ke6 Kxe2 Rc2 Kf1 Rf3 Ke1 Rh3 Re5 Kd6 Rd1 Kc6 Nf2 Rxh4 Rd8 Ra2 Red5 Rh2 Rd2 Rxa5 Rc2 Kb7 Rd7 Kb6 Rd3 Rb5 Rd6 Kb7 Kf1 g5 Kg1 Rh4 Kg2 a5 Nd3 Rg4 Kf3 Rg1 e5 g4 Kf2 Rgb1 e6 Rf5 Kg2 h5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 Nbd7 g4 h6 h4 b5 Rg1 g6 g5 hxg5 hxg5 Nh5 a4 b4 Nc6 Qc7 Nxb4 Bg7 Nd3 Rb8 Ne2 Nc5 Nxc5 dxc5 Rb1 Bd7 c3 Be5 Rg4 Ng3 Bg2 Rh2 Kf2 Nxe2 Qxe2 Bxc3 Bf4 Be5 Bxe5 Qxe5 Kg1 Rh5 Qd2 Bxa4 f4 Qd4 Qxd4 cxd4 Bf1 Bc2 Ra1 Bxe4 Bg2 Bxg2 Rxg2 Rh4 Rf2 Rg4 Kh2 e5 fxe5 Ke7 Kh3 Rxg5 Kh4 Rxe5 Rxa6 Rb7 Raf6 Ke8 Rd6 Reb5 Re2 Kf8 Rxd4 Rxb2 Rd8 Kg7 Ree8 R7b4 Kg3 R4b3 Kf4 Rf2 Kg4 f5 Kg5 Rg2 Kf4 Rg4 Ke5 Re4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 d4 Bg4 Ne5 Bf5 c4 f6 Nf3 c6 Nc3 e6 g3 Bb4 Bg2 Ne7 O-O O-O Qb3 a5 a3 Bxc3 bxc3 Nd7 Nd2 a4 Qa2 Bg6 e4 Bf7 Rb1 Rb8 Qc2 b5 cxd5 cxd5 Qd3 Qa5 Re1 Rfc8 Rb4 Nc6 Rb2 Ne7 Rc2 Nb6 h4 Rb7 Bh3 Rc6 Rb2 Nc4 Rb4 Qc7 Nxc4 Rxc4 Bd2 Qc6 e5 f5 Bf1 Bh5 Qe3 h6 Reb1 Kf7 R1b2 Kg8 f3 Qa6 Rb1 Nc6 Bxc4 dxc4 R4b2 Ne7 d5 Nxd5 Qc5 Bxf3 Rxb5 Rc7 Rb8 Kh7 Qf8 Qa7 Kf1 Re7 R1b2 Kg6 Bc1 Kh5 Ra8 Qc5 Rc8 Qxa3 g4 Bxg4 Rxc4 Qa1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 O-O Nc6 c3 Nf6 Qe2 e6 d4 cxd4 cxd4 d5 e5 Ne4 Be3 Be7 Ne1 f6 f3 Ng5 Nd3 O-O Nd2 Nf7 f4 b6 Nf3 f5 Rac1 Rfc8 Rc2 Rc7 Rfc1 Rac8 Bd2 h6 h3 a5 Kh2 Kh7 Be3 a4 Bd2 a3 b3 Na7 Be1 Rxc2 Rxc2 Nd8 Nb4 Rxc2 Nxc2 Qb5 Qxb5 Nxb5 Bd2 Nc6 Kg1 Kg6 Kf2 Kh5 Ke3 g5 Kd3 g4 Ng1 Ncxd4 hxg4 Kxg4 Nxd4 Nxd4 Be3 Nxb3 Bxb6 Nc1 Kc2 Nxa2 Ne2 Nb4 Kb3 a2 Bd4 h5 Bc3 Nd3 g3 Kf3 Nd4 Kxg3 Nxe6 Nc5 Nxc5 Bxc5 Kxa2 h4 Be1 Kg4 e6 h3 Bf2 Be7 Bg1 Kg3 Kb3 Kg2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d6 d4 Bg4 e4 Nf6 Nc3 e6 Be2 Be7 O-O O-O Be3 Nbd7 Nd2 Bxe2 Qxe2 c5 dxc5 Nxc5 Bxc5 dxc5 Nc4 Qc7 a4 Rad8 e5 Nd5 Ne4 Nb6 Ncd6 f5 Nc3 a6 a5 Nd5 Nc4 Nf4 Qf3 Rd4 b4 Ng6 Rfe1 Rf4 Qe3 Qc6 f3 Bg5 Qe2 Rd4 g3 Rfd8 Rad1 Be7 Qg2 f4 Qe2 h5 Rxd4 Rxd4 Ne4 fxg3 hxg3 h4 c3 Rd5 f4 hxg3 Qg4 Nh4 Nf6 Bxf6 exf6 Nf5 Ne5 Qd6 Qh5 Rxe5 Rxe5 Qd2 Qe2 Qxc3 Rxe6 Qa1 Kg2 gxf6 Qg4 Ng7 Re2 Qd4 Kxg3 Qd3 Kg2 Qb3 Qf3 Qf7 Qe4 c4 Rc2 Kf8 Rxc4 Ne8 Rc8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 Nbd7 g4 h6 h4 Ne5 Rg1 Qb6 Qc1 d5 Be2 dxe4 Nxe4 Nxe4 fxe4 Be7 Rh1 Bd7 c3 Qc7 Bf2 b5 Qc2 Nc4 Bxc4 Qxc4 b3 Qc7 O-O-O O-O Kb1 a5 g5 h5 Rhg1 a4 b4 Rac8 Rd3 g6 Bg3 Qb7 Qe2 Rc4 Be5 Rfc8 Qe3 Bc6 Re1 a3 Qf2 Be8 Nb3 Qc6 Na5 Rxb4 Ka1 Qc5 cxb4 Qxb4 Qd2 Rc2 Qxb4 Bxb4 Rd8 f6 Rf1 fxe5 Rxe8 Kg7 Nb3 Bc3 Kb1 Rb2 Kc1 Rxb3 Kc2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Qc2 O-O Bd3 Nbd7 Nge2 Re8 O-O Nf8 a3 g6 b4 Ne6 Bh4 a6 f3 Ng7 Bf2 h5 h3 Nh7 e4 Bh4 Qd2 Bxf2 Rxf2 h4 Bc2 Nh5 Nf4 Nxf4 Qxf4 Qf6 Qd2 Be6 e5 Qg7 Na4 Nf8 Qg5 Qh8 f4 f6 exf6 Bf7 Bxg6 Bxg6 f5 Kf7 Nc5 Bh5 Nxb7 Nd7 Nd6 Kf8 Qxh4 Nxf6 g4 Re7 g5 Rd7 gxf6 Rxd6 Re1 Re8 Rxe8 Kxe8 Re2 Kf8 Re6 Qh6 Rxd6 Qe3 Qf2 Qxh3 Qf4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 d5 cxd5 Nxd5 Ne2 cxd4 exd4 Nc6 O-O Nf6 Bc2 Bd6 a3 O-O Qd3 h6 Rd1 Bd7 Bf4 Bxf4 Nxf4 Re8 Re1 Ne7 Ne4 Nxe4 Qxe4 Nf5 Rad1 Rc8 Bb1 Ba4 Rd2 Qa5 b4 Qc7 Rde2 Bb5 Rc2 Qd7 Rxc8 Qxc8 g4 Qd8 Rd1 Qg5 Ng2 Nd6 Qh7 Kf8 h3 Bc6 Ne3 Ke7 Qc2 Rc8 Re1 Bf3 Qd3 Kf8 d5 Bxd5 Nxd5 Qxd5 Qxd5 exd5 Rd1 Ke7 Kg2 Rc3 Rxd5 Rxa3 Rd4 Rb3 Ba2 Rb2 Bd5 Kf6 Bf3 Ke5 Rd5 Ke6 Rd4 b6 Kg3 a5 bxa5 bxa5 Ra4 Rb5 h4 g5 hxg5 hxg5 Be2 Rc5 f4 f6 Kf3 Rc3 Kf2 gxf4 Rxf4 Ke5 Ra4 Ne4 Kg2 Rc5 Kf3 Nc3 Ra3 Kd4 Bf1 Re5 Kg3 Re3 Kh4 Re5 Bg2 Kc4 Ra1 a4 Rf1 Re6 Rf4 Kb3 Bf1 Kc2 Bc4 Re1 Rxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O O-O c5 d4 d6 Nc3 Nc6 dxc5 dxc5 Bf4 Be6 Ne5 Na5 Nd3 Nh5 Nxc5 Nxf4 gxf4 Bxc4 Rc1 Qc7 Nxb7 Qxf4 e3 Qe5 Nxa5 Qxa5 Bxa8 Bxf1 Qd5 Qxd5 Nxd5 Rxa8 Kxf1 Bxb2 Nxe7 Kf8 Rc8 Rxc8 Nxc8 a6 Nb6 Ke7 Ke2 Ke6 Nc4 Be5 h3 f5 f4 Bc7 Kd3 h6 Nd2 g5 fxg5 hxg5 e4 Kf6 exf5 Kxf5 Kd4 Kf4 Ne4 Bd8 Kd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Bf4 Re8 Rad1 Ne4 Nb5 Qf6 d6 Bd7 g3 g5 Be3 h6 Nc7 Nxc7 dxc7 Bc6 Nd2 Rac8 Nxe4 Rxe4 Qxc5 Rxc7 Rd6 b6 Rxf6 bxc5 Rxc6 Rxc6 Bf3 Rxe3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c6 Nf3 Bg7 Bg2 d5 cxd5 cxd5 Nc3 O-O Ne5 e6 O-O Nfd7 f4 Nc6 Be3 Nb6 Bf2 Bd7 e4 Ne7 Nxd7 Qxd7 e5 Rfc8 Rc1 Bf8 Bf3 Rc7 b3 Rac8 Qd2 Nc6 Qb2 a6 Be2 Qe7 Nb1 Nb4 Nc3 Nc6 Nb1 Nb4 Rc5 Nd7 Rxc7 Rxc7 Nc3 Nc6 Nb1 Nb4 Nc3 Nc6 Nb1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 Bb4 Bg2 O-O O-O e4 Ng5 Bxc3 bxc3 Re8 f3 e3 d3 d5 Qb3 Na5 Qa3 c6 cxd5 cxd5 f4 Nc6 Rb1 Qc7 Bb2 Bg4 c4 dxc4 Bxf6 gxf6 Ne4 Kg7 dxc4 Rad8 Rb3 Nd4 Rxe3 Qxc4 Kh1 Nf5 Rd3 Bxe2 Rxd8 Rxd8 Re1 Re8 Qa5 b5 Nd2 Qd3 Nb3 Bf3 Bxf3 Qxf3 Kg1 Rxe1 Qxe1 Ne3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c6 Bg2 d5 cxd5 cxd5 Nf3 Bg7 Nc3 O-O Ne5 e6 O-O Nfd7 f4 Nc6 Be3 Nb6 Bf2 Ne7 a4 a5 Qb3 Bd7 Rfc1 Bc6 Nb5 Nbc8 e3 Nd6 Nxd6 Qxd6 Be1 Rfb8 Bf1 f6 Nf3 Qd7 Qc2 Nf5 Bd2 Nd6 b3 Rc8 Qd1 h6 Be1 g5 Ra2 Qe8 Rac2 Bf8 Bd3 g4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 Bb4 Bg2 O-O O-O e4 Ng5 Bxc3 bxc3 Re8 f3 exf3 Nxf3 Qe7 e3 Ne5 Nd4 Nd3 Qe2 Nxc1 Raxc1 d6 Rf4 c6 Rcf1 Qe5 Qd3 Bd7 Nf5 Bxf5 Rxf5 Qe6 Qd4 Re7 Qh4 Nd7 Bh3 Nf8 R5f3 Qe5 d4 Qe4 Qxe4 Rxe4 Rxf7 Rxe3 d5 Rae8 Rxb7 cxd5 cxd5 R3e7 Rfb1 h5 a4 g5 Bf5 Kg7 a5 Kf6 Bd3 Rxb7 Rxb7 Re3 Bb5 Rxc3 Rxa7 Ng6 Rd7 Ne5 Rxd6 Kf5 a6 Ra3 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 e5 Qd5 Bf2 Rf8 Kg1 Bh6 h4 Qf7 Bg3 Be3 Kh2 Qc4 Rb1 b6 Rb2 Qd5 Qd3 Nc4 Rb1 b5 Kh3 a6 Ng1 cxd4 Nf3 Rd8 a4 dxc3 Qxc3 Qe6 Kh2 bxa4 Rb4 Nd2 Rxa4 Nf1 Kh3 Rd1 Qc2 Rc1 Qe2 h5 Be1 Qd7 Qxa6 Ra1 Qxg6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nc6 g3 g6 Bg2 d6 Rb1 Bf5 d3 Qd7 b4 Bg7 b5 Nd8 Nd5 c6 bxc6 bxc6 Nc3 Ne7 Ba3 O-O Nf3 h6 O-O Be6 Qa4 f5 Nd2 Nf7 Rb3 Rfb8 Rfb1 Rxb3 Rxb3 Rc8 Qa6 Nd8 Bb4 Rb8 Na4 Kf7 Bc3 Rxb3 axb3 Qc7 e3 Bc8 Qa5 Qxa5 Bxa5 Ne6 Bb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 e5 Qd5 Bf2 Rd8 Qe1 Qe4 g5 Qf5 h4 Nc4 Kg1 Qg4 a4 h6 Ra2 hxg5 Qb1 gxh4 Qb3 Qe6 Nf4 Qf7 Nxg6 Qxg6 Qxc4 Kh8 Rb2 cxd4 cxd4 Qg4 Qf7 Rxd4 Bxd4 Qxd4 Rf2 Qxe5 Rf5 Qe1 Rf1 Qe5 Kh1 b6 Qf4 Qh5 Qf5 Qe2 Rc1 Bf6 Qg6 Qe6 Rd1 Qc8 Rf1 Qd7 Qh5 Kg7 Rf4 Qd2 Rg4 Kf8 Qf5 Qc1 Kh2 Qc7 Qf4 Qxf4 Rxf4 Ke8 Kg1 a6 Kf2 Kd7 Ke2 Kd6 Kd3 Kc5 Rc4 Kd5 Rc7 a5 Rc4 e5 Rg4 Be7 Rg7 e4 Ke3 Bc5 Ke2 Bd4 Rg5 Kc4 Rf5 Kc3 Rh5 Kc4 Rf5 Kc3 Rg5 Kc4 Rh5 Bf6 Rh6 Bd4 Rh5 Bf6 Rh6 Bd4 Rxh4 b5 axb5 a4 Rxe4 a3 b6 a2 Rxd4 Kxd4 b7 a1=Q b8=Q Qa6 Kf2 Qf6 Kg1 Ke4 Qb4 Kf5 Qe1 Qd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 d6 g3 c5 Bg2 Nc6 a3 g6 b4 Bg7 Rb1 Nge7 e3 O-O d3 Rb8 Nge2 Be6 b5 Na5 Bd2 b6 O-O Nb7 e4 Kh8 Qc1 f5 Bg5 Qe8 Bxe7 Qxe7 exf5 Bxf5 Nd5 Qd7 Qd2 Na5 Nec3 Rbe8 Ne4 Nb7 a4 Na5 h4 Nb7 Kh2 Rb8 Ra1 Na5 Ra3 Rf7 Qc3 Rd8 Ra2 Bh6 Ng5 Rff8 Re2 Bg7 Qc2 Rde8 Ne3 Bh6 Bd5 Bg7 Qd1 h6 Ne4 Qd8 Ra2 Bc8 Nc3 h5 Be4 Re6 Ncd5 Bh6 Ng2 Kg7 f4 exf4 Ngxf4 Re5 Nxg6 Rxf1 Qxf1 Rxe4 dxe4 Kxg6 Rf2 Qe8 e5 dxe5 Rf6 Kg7 Rd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 cxd4 cxd4 Qb6 Kg1 Qe6 Qd3 Qxg4 Rf1 Rc8 h3 Qd7 d5 Nc4 Bd4 e5 dxe6 Qxe6 Bxg7 Kxg7 Nf4 Qd6 Qc3 Kh6 Nd5 Qe5 Qd3 Kg7 Nf6 Qd6 Qc3 Qe5 Qd3 Qd6 Qc3 Qe5 Qb3 Rc7 Qd3 Rf7 Qxc4 Rxf6 Rd1 b5 Rd7 Kh6 Qe2 Qc5 Kh2 Qe5 g3 Qc3 Kg2 Qc4 Qe3 g5 Rd2 Qf1 Kh2 Qf3 Qd4 Re6 e5 Qf5 Re2 a5 Qd5 b4 Qxa5 Qd3 Rg2 Qd4 Qa8 Qxe5 Qf8 Kg6 Qxb4 h5 h4 gxh4 Qxh4 Rd6 Qc4 Rd4 Qc6 Kg7 Qb7 Kh6 Qc6 Kg7 Rc2 Rh4 Kg2 Qe4 Qxe4 Rxe4 Rc7 Kg6 Ra7 Re3 Kh3 Rc3 Ra8 Rc4 a4 Kg5 a5 Ra4 a6 Kh6 Kg2 Ra3 Kf2 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6 Nxf6 c3 Bg4 h3 Bxf3 Qxf3 e6 Bc4 Be7 O-O Nd5 Be3 Qb6 Qe2 O-O Rad1 Bd6 Bb3 Nxe3 fxe3 c5 Rf3 Rae8 Rdf1 Re7 Qf2 Qc7 Qh4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 Kg1 Qe6 Qd3 Qc4 Qxc4 Nxc4 Bf2 cxd4 cxd4 e5 d5 Bh6 h4 Bd2 Rd1 Ba5 Rc1 b5 Rc2 Nd6 Ng3 Nc4 Nf1 Nd6 Ng3 Nc4 g5 Kf7 Nf1 Nd6 Ng3 Nc4 Kf1 Ke7 Bc5 Kf7 Rf2 Kg7 Rf6 Bb6 Rc6 Na5 Bxb6 Nxc6 Bc7 Rf8 Ke2 Rf7 Bd6 Rd7 Bc5 Na5 Nf1 Rc7 Bd6 Rc2 Kd3 Rxa2 Ne3 Kf7 Ng4 Nc4 Nxe5 Nxe5 Bxe5 b4 Bf6 b3 e5 Rxg2 e6 Kf8
Result: 0-1
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 cxd5 exd5 Bf4 Nf6 e3 Bf5 Nge2 O-O Rc1 c6 Ng3 Be6 Bd3 Re8 Qb3 Qb6 Qc2 Nbd7 O-O g6 h3 Bf8 Nge2 Rac8 Qd2 Nh5 Bh2 Ng7 g4 Qd8 f3 Nb6 b3 Ba3 Rc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 cxd5 Nxd5 e4 Nxc3 bxc3 c5 Rb1 O-O Be2 cxd4 cxd4 Qa5 Qd2 Qxd2 Bxd2 e6 O-O b6 Rfd1 Bb7 d5 exd5 exd5 Nd7 Bb4 Rfc8 Be7 Bf6 d6 Kg7 Re1 Rc5 Bb5 Bc6 Bxc6 Rxc6 Rbd1 Bc3 Re3 f6 g4 g5 h4 h6 hxg5 hxg5 Nd4 Bxd4 Rxd4 Rh8 Re1 Rc2 a4 a5 f4 Kg6 fxg5 Kxg5 Rf1 Kg6 Rf2 Rhc8 Rdf4 Rxf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6 Nxf6 c3 Bg4 h3 Bxf3 Qxf3 Qd5 Be2 e6 O-O Bd6 Qd3 Bc7 Bf3 Qd7 Rd1 O-O c4 Rad8 Qb3 Qe7 g3 Bb8 Be3 Rd7 Rd2 Rfd8 Rad1 h6 a3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Rd1 Re8 d6 h6 h3 Nb4 Bf4 Nd7 Rd2 a6 Qb3 b5 Qd1 c4 a4 Nc5 axb5 Nbd3 Bxd3 Nxd3 Rxd3 cxd3 Nd5 axb5 Ne7 Kh7 Rxa8 Qxa8 Nxf5 gxf5 Qxd3 Qe4 Qxb5 Ra8 Bd2 Rd8 Qc5 Qe6 Bf4 Bxb2 Nh4 Bf6 Qxf5 Qxf5 Nxf5 h5 g4 hxg4 hxg4 Kg6 Kg2 Bb2 Ne7 Kf6 Nc6 Rd7 Nb8 Rd8 d7 Ke6 Kf3 Ba3 Bc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 Bb4 Bg2 O-O O-O Re8 d3 Bxc3 bxc3 e4 Nd4 h6 dxe4 Nxe4 Qc2 d5 cxd5 Qxd5 e3 Na5 f3 Nd6 e4 Qc5 Be3 Ndc4 Bf2 Qe7 Rad1 Bd7 f4 Rad8 e5 Bg4 Nf5 Qe6 Rxd8 Rxd8 Nd4 Qc8 f5 c5 Qe4 cxd4 Qxg4 Nxe5 Qe2 Nec6 cxd4 Nxd4 Bxd4 Rxd4 f6 Qe6 Qb2 Qe3 Kh1 b6 fxg7 Nc4 Qc2 Kxg7 Bd5 Nd6 Qb2 Qe5 Bb3 a5 Qf2 f5 Qb2 b5 a3 Kg6 Qf2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 b3 c5 a3 Ne8 Rb1 f5 b4 axb4 axb4 b6 Qb3 Nf6 Bd3 Bh6 Rb2 Ra1 Qc2 Bf4 Nf3 fxe4 Nxe4 Nxe4 Bxe4 Rxc1 Rxc1 Bxc1 Qxc1 Nf5 Qg5 Nd4 Qxd8 Nxf3 Bxf3 Rxd8 bxc5 bxc5 Rb8 Rf8 Rb6 Rf6 Rb8 Rf8 Rb6 Rf6 Be4 Bf5 Bxf5 Rxf5 g3 Rf6 h4 h6 Kg2 Kg7 f3 Kg8 Kf2 g5 hxg5 hxg5 Ke3 Kg7 Rb8 Kh7 Rd8 Kg7 Ra8 Kf7 Ke4 Kg7 Ra7 Kg6 Re7 g4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 Nf3 Nf6 Bg5 h6 Bh4 O-O e3 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qd3 Rfc8 Rfc1 Rab8 h3 g6 Bxc6 Rxb1 Qxb1 Qxc6 dxc5 Qxc5 Ne2 Qf8 h4 Ne5 Nxe5 Bxe5 Rd1 Qc5 h5 Qc2 Qxc2 Rxc2 Rxd5 Rxe2 Rxe5 Rxa2 hxg6 fxg6 Re7 a5 Ra7 a4 g3 h5 Kg2 a3 e4 g5 Kf3 g4 Ke3 Ra1 Kf4 Rf1 Kg5 Rxf2 Kxh5 Re2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 O-O Nd7 b4 c5 bxc5 bxc5 Qb3 cxd4 Nxd4 Bxd4 exd4 Nb6 a4 Rb8 a5 Nc4 Bxc4 dxc4 Qxc4 Qd6 Qc5 Qxc5 dxc5 Rbc8 a6 Ba8 Nb5 Rxc5 Nxa7 Be4 f3 Ra8 fxe4 Rxa7 Ra4 Rc6 Rfa1 Kf8 Kf2 Ke7 Ke3 Ke6 Ra5 Rd6 R1a2 Rc6 h4 Rd6 Kf4 Rb6 R2a3 Rc6 Re5 Kf6 Rf5 Ke6 Rfa5 Rb6 Re5 Kf6 Rea5 Ke6 Ra1 Rc6 Re5 Kf6 Rf5 Ke6 Re5 Kf6 Rea5 Ke6 R1a2 Rb6 g4 f6 h5 Rc6 Rb2 Rcxa6 Rb6 Rxb6 Rxa7 Rb1 Rxg7 Rf1 Ke3 Re1 Kf3 Rf1 Ke2 Rf4 Ke3 Ke5 Re7 Kd6 Re6 Kxe6 Kxf4 Ke7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 Nf3 Nf6 Qc2 O-O Bg5 c5 dxc5 dxc4 e4 Qa5 e5 Nd5 Bxc4 Nxc3 O-O Qxc5 Qxc3 Nc6 Bxe7 Qxe7 a3 Bd7 Rac1 Rfd8 b4 a6 Qe3 Be8 Bd3 Na7 Bb1 Bc6 Ng5 h6 Ne4 Nb5 Rc4 Bxe4 Bxe4 Rac8 Rxc8 Rxc8 Rc1 Rxc1 Qxc1 Qd7 g3 b6 Kg2 Qd8 h4 a5 bxa5 bxa5 Qc5 Nd4 h5 f5 Bb7 Kf7 Kh2 Qb8 Qxd4 Qxb7 g4 Qf3 Qd7 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Rd1 Re8 d6 h6 Bf4 Nd7 Rd2 Nb4 Qb3 Be6 Bc4 Nb6 Bxe6 Rxe6 a3 Nd3 Bg3 c4 Qc2 Rc8 Rad1 Qd7 h4 f5 Rxd3 cxd3 Qxd3 Nc4 Qd5 Nb6 Qd3 Nc4 Qd5 Nb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Rc1 Nc6 cxd5 exd5 Be2 d4 exd4 Nxd4 Nxd4 Qxd4 Qxd4 Bxd4 Nb5 Bb6 O-O Be6 a3 Rfd8 Nd6 Rd7 Bb5 Re7 Rfe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 d5 cxd5 Nxd5 d4 Nxc3 bxc3 g6 e3 Bg7 Bd3 O-O O-O Qc7 Rb1 b6 Qe2 Rd8 Be4 Ba6 c4 Nc6 d5 f5 Bd3 e5 e4 Nd4 Nxd4 cxd4 Bg5 Rf8 Rfc1 Rac8 Bd2 Rf7 a4 fxe4 Qxe4 Rcf8 f3 Bc8 a5 Bf5 Qe2 Re8 Be4 Bf8 Qd3 Bc5 Ra1 Qd7 Re1 Qc8 Kh1 Rc7 Rab1 Kg7 Rec1 Bxe4 fxe4 Rf7 Qg3 bxa5 Bxa5 Rf4 Re1 Qa6 Bd2 Rf7 Qd3 Ref8 h3 Rf2 Ra1 Qf6 Rg1 h5 Ra5 Qe7 Rb1 h4 Ra6 R8f7 Rc6 Qf8 Rg1 Be7 Re6 Kh7 Be1 Rf1 Bd2 Bc5 Rc6 R7f3 gxf3 Rxf3 Rc7 Kh8 Bh6 Rxd3 Bxf8 Rxh3 Kg2 Rg3 Kh2 Rxg1 Bxc5 d3
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nf3 Nf6 g3 d5 b3 Be7 Bg2 O-O O-O b6 Bb2 Bb7 e3 Nbd7 Nc3 Ne4 Ne2 a5 d3 Bf6 Qc2 Bxb2 Qxb2 Nd6 cxd5 Bxd5 d4 c5 Rfd1 Rc8 Nf4 Bxf3 Bxf3 Qe7 Rac1 Rfd8 dxc5 Nxc5 b4 axb4 Qxb4 Qa7 a3 Nf5 Rb1 Rxd1 Rxd1 Qc7 Nd3 h6 Rc1 Ne7 Qb5 Nf5 a4 Nd6 Qb1 Qa7 Ne5 Nxa4 Rxc8 Nxc8 Qd1 Ne7 Qd8 Kh7 Nxf7 Ng6 Qe8 Qe7 Qxa4 Qxf7 Be4 Kg8 Qb5 Nf8 Qxb6 Qf6 Qb5 Qe7 Kg2 g6 Qa5 Qg7 Qc5 Qf7 h4 h5 Qc6 Qe7 Bd3 Qf7 Qd6 Kg7 e4 Kg8 Bc4 Kg7 Qe5 Kg8 Qd6 Kg7 Bb5 Kg8 Bc6 Qa7 Qb4 Qc7 Qb7 Qd8 e5 Qa5 Be8 Qc5 Qf7 Kh8 Ba4 Qd5 Kh2 Qc5 Bb3 Qc8 Bd1 Qc5 Kg2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Qa5 Bd2 Qd8 Bg5 Qa5 Bd2 Qd8 c4 Nxe4 cxb5 Be6 Bc4 Ne7 Be3 Rc8 Nb6 d5 Nxc8 Nxc8 O-O Bxa3 bxa3 dxc4 Qxd8 Kxd8 bxa6 Kc7 Rab1 Ncd6 Rfc1 Ra8 f3 Nf6 a7 Nd7 Rb4 Nf5 Rbxc4 Bxc4 Rxc4 Kd8 Bf2 Nd4 Ra4 Kc7 Kf1 Nc5 Rc4 Kd6
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Nh3 h5 Nd5 Bg4 f3 Bxh3 Bxh3 e6 Ne3 h4 f4 hxg3 hxg3 d5 exd5 exd5 Bg2 Rxh1 Bxh1 Nf6 Bd2 Qe7 Kf2 O-O-O Qf1 c4 Bf3 cxd3 cxd3 Ne4 dxe4 dxe4 Bg4 f5 Rd1 fxg4 Bc3 Bxc3 bxc3 Rh8 Qc4 Kb8 Qd5 Ka8 Qd6 Qh7 Qe6 Ne7 Qxe4 Nf5 Nxf5 gxf5 Qd5 a6 Re1 Rc8 Re6 Qh2 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 g3 d5 Bg2 Be7 O-O O-O d4 c6 Qc2 Nbd7 Nbd2 b6 e4 Ba6 b3 Rc8 Bb2 Qc7 Rfe1 Qb8 Rac1 Rfd8 Qb1 h6 a3 Bb7 e5 Ne8 cxd5 cxd5 Bf1 Rxc1 Rxc1 Rc8 b4 Rxc1 Bxc1 Qc8 b5 Qc3 a4 g5 h3 Nf8 Bb2 Qc8 Bd3 Ng7 Nf1 Qd8 Ne3 h5 Qc1 Ng6 Ba3 Bxa3 Qxa3 f5 exf6 Qxf6 Bxg6 Qxg6 Qd6 g4 hxg4 hxg4 Ne5 Qb1 Nf1 Qc1 Nxg4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d6 c4 Nbd7 Nc3 c6 e4 e5 Be2 Be7 O-O O-O Re1 a6 Bf1 b5 a3 Bb7 Bg5 h6 Bh4 Re8 Rc1 Qa5 c5 exd4 Nxd4 Qc7 Nf5 Nxc5 Bxf6 Bxf6 b4 Nd7 Nd5 Qb8 Nxf6 Nxf6 Qxd6 Re6 Qxb8 Rxb8 e5 Nd7 f4 f6 Nd4 Ree8 e6 Nf8 f5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Bd3 Bb7 f3 Nh5 Nh3 f5 Bd2 O-O Qe2 c5 d5 Bxc3 Bxc3 exd5 O-O-O Qe8 Rhg1 dxc4 Bxc4 Kh8 g4 fxg4 Rxg4 Nf6 Rdg1 Nxg4 Rxg4 Rf6 Bxf6 gxf6 Rg8 Qxg8 Bxg8 Kxg8 Qc4 Kf8 Qh4 Kg7 Nf4 Nc6 Nh5
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 g3 e6 Nf3 d5 cxd5 exd5 d4 Nc6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8 Rc1 Bg4 h3 Be6 Kh2 Qd7 Nxc6 bxc6 Na4 Bf5 Bc5 Bd8 Bd4 Ne4 f3 Ng5 Nc5 Qe7 e4 dxe4 fxe4 Bg6 e5 Bb6 h4 Rad8 hxg5 Qxg5 Qa4 Qh5 Kg1 Rxe5 Bxe5 Qxe5 Qb4 Rd3 Rf2 Rxg3 Qf4 Rxg2 Kxg2 Qd5 Kg3 Bxc5 Rd2 Qh5 Qg4 Qe5 Qf4 Qe7 Rc3 Bb6 Kg2 Qe1 Rc1 Qe6 Qg3 Be4 Kh2 Qf5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bxf6 Bxf6 Qb3 c6 e3 O-O Rd1 Qb6 Qc2 dxc4 Bxc4 c5 Ne4 Be7 dxc5 Bxc5 O-O Be7 Be2 Bd7 Ne5 Rc8 Qd3 Be8 Nc4 Qc7 Ned6 Rd8 Nxe8 Rxe8 Bf3 Rd8 Qb3 Nc6 g3 Bf6 Kg2 Rab8 Rxd8 Rxd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Ba6 Qc2 Bb7 Nc3 c5 e4 cxd4 Nxd4 Bc5 Nb3 Nc6 Bg5 a6 O-O-O Qc7 Kb1 O-O-O Qd2 d6 f3 h6 Bf4 Ne5 h4 Kb8 h5 Rd7 Rc1 Rc8 Be2 Ka7 Rhd1 Rdd8 g4 Ng8 Bg3 Ne7 f4 N5c6 Bf3 Rb8 Bh4 Rd7 Nb5 axb5 cxb5 Na5 Nxa5 bxa5 Rxc5 dxc5 Qxd7 Qxf4 Rd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nc3 Nc6 e3 O-O O-O Re8 Re1 e5 d5 Na5 e4 c6 Bg5 f6 Be3 Nac4 dxc6 Nxe3 Qxd8 Rxd8 cxb7 Bxb7 Rxe3 Bh6 Ree1 Nc4 Rad1 Kf8 h4 Rac8 Bh3 Rxd1 Rxd1 Nxb2 Rd7 Rxc3 Rxb7 Nc4 Nh2 Nd6 Rxh7 Bg7 h5 gxh5 Rxh5 Rc1 Kg2 Rc2 Be6 Nxe4 Ng4 Rd2 Bb3 a5 Rf5 Nd6 Rh5 Ne4 Rf5 Nd6 Rh5 Ne4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Nf3 Ngf6 Ng3 e6 Bd3 Be7 O-O c5 Qe2 O-O Rd1 Qc7 c4 cxd4 Nxd4 a6 b3 Re8 Bb2 b6 Nh5 Bb7 Nxe6 fxe6 Qxe6 Kf8 Bxh7 Nc5 Qh3 Nxh7 Bxg7 Kg8 Bb2 Qc6 Rd4 Ne4 Re1 Neg5 Qg4 Ba3 Bc3 Rxe1 Bxe1 Re8 Bd2 Bc1 h4 Bxd2 Rxd2 Re1 Kh2 Re4 f4 Qe6 Rd8 Kf7 Rd7 Kf8 Qxe6 Rxe6 hxg5 Re7 Rxe7 Kxe7 g4 Be4 Kg3 Bb1 a3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Qa4 Qxa4 Nxa4 Ne4 f3 Bd7 fxe4 Bxa4 cxd5 Bxb2 Rb1 Bc3 Kf2 Nd7 Rc1 Nxc5 Kf3 Bb4 Rc4 Bb5 Rxb4 Bxf1 Ne2 Bxe2 Kxe2 f5 exf5 gxf5 Rd1 Rd8 Kf3 Rg8 Be5 Rd7 Rh4 e6 Rc1 Rxd5 Bd4 Ne4 Rc8 Rd8 Rxd8 Kxd8 g4 Nd2 Ke2 Ne4 Kf3 Nd2 Ke2 Ne4 gxf5 exf5 Kf3 Ng5 Kf4 Ne6 Kxf5 Nxd4 exd4 Rg7 d5 Kd7 Rh6 Rg2 a4 Rf2 Ke5 Re2 Kd4 Rd2 Kc4 Rc2 Kb3 Rd2 Rxh7 Kd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be2 Be7 O-O O-O f4 a6 Kh1 Qc7 a4 Rd8 Qe1 b6 Qg3 Bb7 f5 e5 Bh6 Bf8 Bg5 Be7 Bh6 Bf8 Bg5 Be7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 dxc4 a4 Bf5 Nh4 Bc8 Nf3 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qb3 a5 Na2 Be7 Nh4 Bg6 g3 Qc7 Nc3 O-O Nxg6 hxg6 Rd1 Bb4 Qc2 Rad8 Na2 Be7 Bd2 Qb6 Rac1 Nd5 e4 N5f6 Be2 e5 Be3 exd4 Bxd4 Qc7 f4 g5 e5 Nd5 Qe4 N7b6 Bd3 g6 f5 f6 fxg6 f5 Qe2 Kg7 Qh5 Rh8 Qf3 Nf4 Bxf5 Rxd4 Rxd4 Qxe5 Re4 Bc5 Kh1 Qxf5 gxf4 Bd6 Qc3 Kxg6 Qd3 Be7 fxg5 Qd5 Qe2 Rh4 Nc3 Rxe4 Nxe4 Nxa4 Rd1 Qe6 Qc2 Qf5 Qxa4 Qf3 Kg1 Qg4 Kf2 Qf4 Ke2 Qg4 Kd3 Bb4 Qc2 Qf3 Kd4 Kg7 Ke5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 Qc2 c5 dxc5 Bxc5 Bg5 Qc8 Rc1 h6 Bh4 a5 Na4 Nd7 e4 Nc7 Nxc5 bxc5 Be2 Ba6 O-O O-O Rfd1 f6 Bg3 Rf7 Bc4 Bxc4 Qxc4 e5 Nh4 Qe8 Nf5 Qe6 Qe2 Rb8 Rd6 Qe8 Rcd1 Nf8 f4 Nb5 fxe5 Nxd6 Nxd6 Qa4 Nxf7 Rxb2 Nxh6 Kh7 Qh5 g6 Qf3 Kxh6 Rf1 Qd4 Kh1 Nh7 exf6 Nxf6 Bf4 Kg7 Bg5 Rb6 Qh3 Kg8 Qc8 Kg7 Qc7 Nd7 Qf4 Qc4 h3 Qe6 Rd1 Rc6 Bd8 Nb6 Rf1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 e5 Qd5 Bf2 Rd8 Qc2 Qc4 Qb2 Bh6 h4 Qf7 Kg1 Rf8 Ng3 Nc4 Qe2 Qxf2 Qxf2 Be3 Qxe3 Nxe3 dxc5 Rc8 Rb1 Rxc5 Rxb7 Nxg4 Rb4 h5 Ne4 Rxe5 g3 Kf7 Kg2 Ke6 Kf3 Kf5 a4 Rd5 Nf2 Nxf2 Kxf2 Rd3 Rc4 e5 a5 Rd5 a6 Rd6 Ra4 Rc6 Ra3 Ke4 Ke2 Rc4 Kf2 Kd3 Ra5 Rc6 Rxe5 Rf6 Kg2 Kxc3 Re7 Rxa6 Rg7 Kd4 Kh3 Ke5 Kg2 Ra2 Kf3 Ra3 Kg2 Kf6 Rc7 a5 Rc6 Kf5 Rc5 Ke4 Rg5 Ra2 Kh3 a4 Rxg6 a3 Ra6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Qa4 Qxa4 Nxa4 O-O Nf3 Ne4 Be5 Bd7 Nc3 Nxc3 bxc3 dxc4 Bxc4 Rc8 Bd5 Bc6 Bxc6 Rxc6 Bxg7 Kxg7 Rb1 Rc7 Nd4 Na6 c6 b6 f4 Nb8 Rb4 Nxc6 Rc4 Rac8 Nb5 Rd7 Nd4 Rdc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 O-O h6 Ne4 Nxe4 Bxe4 O-O c3 e5 Bc2 Re8 Re1 exd4 Rxe8 Qxe8 Qxd4 Qe7 Bf4 Bxf4 Qxf4 Nf8 Re1 Be6 Nd4 Rd8 h4 Qc5 Re3 Qd6 Nxe6 fxe6 Qg4 Qd2 Bb3 Kh8 Re2 Qd6 g3 a6 Kg2 Re8 Re3 Re7 Rf3 Rd7 Qh5 Qe7 Qe5 Rd8 a4 b5 Qe4 Qc7 Rf4 c5 Qf3 Qd6 axb5 axb5 Rf7 Rb8 Ra7 b4 Bc2 bxc3 bxc3 Qe5 Rf7 Nh7 Qg4 Kg8 Re7 Nf8 Qf3 c4 Be4 Kh8 Bc6 Nh7 Qf7 Nf8 Re8 Rxe8 Bxe8 Nh7 Bd7 Nf6 Bxe6 h5 Bxc4 Qe4 Kh2 Kh7 Qe6 Qf3 Qe1 Ng4 Kg1 Qc6 Bd3 g6 Qe7 Kh6 Be4 Qb6 Qf8 Kh7 Qf7 Kh6 c4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 a4 Nc6 Be2 e6 O-O Be7 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Bd7 Nb3 Na5 Nxa5 Qxa5 Qd3 Rad8 Qd2 Rc8 e5 dxe5 fxe5 Qxe5 Bxb7 Rcd8 Bf4 Qa5 Qe2 Bc8 Bc6 Rf8 Bf3 Qb4 Bc7 Rd7 Be5 Bb7 a5 Rfd8 Ra4 Bxf3 Qxf3 Qc5 Bxf6 gxf6 Rh4 f5 Rh3 Qc4 g4 f4 g5 Rd4 Rf2 Bxg5 Qg4 f6 Rhf3 Kh8 Rg2 Rd2 h4 Rxg2 Kxg2 Bh6 Kh3 Rg8 Qh5 Qc6 Kh2 Bf8 Rf2 f5 Qe2 Rg4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 f4 d5 exd5 Nf6 c4 e6 dxe6 Bxe6 Nf3 Nc6 Nc3 Bd6 d4 cxd4 Nxd4 O-O Nxe6 fxe6 Be3 Qe7 Qf3 Bb4 Be2 e5 O-O-O Bxc3 bxc3 exf4 Bd4 Rae8 Bxf6 Rxf6 Rd2 Kh8 Rhd1 Qa3 Rb2 Ne5 Qh3 f3 gxf3 Rb6 Rd2 Nd3 Bxd3 Re1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 a6 O-O Be7 f4 O-O Be3 Qc7 a4 b6 Bf3 Bb7 Qe2 Nbd7 Bf2 g6 e5 dxe5 fxe5 Ne8 Rae1 Ng7 Kh1 Rad8 Bxb7 Qxb7 Nf3 Nc5 Bxc5 bxc5 Ne4 Nf5 b3 h5 Ned2 Qb4 Nc4 Rd5 Qf2 Rfd8 Re4 Qb7 Ne3 Nxe3 Qxe3 Rd1 Kg1 Qd5 Re1 Rd7 Qe2 Rxe1 Nxe1 Qd2 Nd3 Qxe2 Rxe2 c4 bxc4 Rd4 Nb2 Bc5 Kf1 Rf4 Ke1 Bb4 c3 Bxc3 Kd1 Bxb2 Rxb2
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 g3 Bg7 Bg2 O-O d4 c5 Nf3 cxd4 Nxd4 Nc6 Nc2 d6 O-O Rb8 b3 Ng4 Bd2 a6 Rb1 Bd7 a4 f5 e3 Kh8 Qe2 e5 Nd5 g5 Rbd1 f4 h3 Nh6 g4 Nf7 e4 b5 axb5 axb5 Ra1 bxc4 bxc4 Qc8 Rfb1 Rxb1 Rxb1 Qa6 Qd3 Rc8 Na3 Be8 Nb5 Nb8 Nbc7 Qa2 Nxe8 Rxe8 Rb7 Qa1 Kh2 Rf8 Ne7 Qa6 Qd5 Bf6 Rxb8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Nc6 e3 e5 Nc3 Bg4 d5 e4 Nxe4 Qxd5 Qxd5 Nxd5 h3 Bxf3 Bxf3 O-O-O Nc5 b6 Nb3 a5 e4 Ndb4 Bg5 a4 Bxd8 Rxd8 Rad1 axb3 Bg4 Kb7 Rxd8 Nxd8 axb3 Bxb2 Rd1 Ndc6 Be2 Nd4 Bc4 f6 Kg2 c5 h4 b5 Bg8 h6 Bf7 g5 hxg5 hxg5 f4 g4 Rh1 Ne2 Rh7 Kb6 Kf2 Nc1 Rh6 Kc7 Rh5 Bd4 Kg2 Kd6 Bg8 Ne2
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 g3 Bg7 Bg2 O-O d4 d6 Nf3 Nbd7 O-O e5 e4 c6 h3 Re8 Re1 exd4 Nxd4 a5 Rb1 Nc5 Bf4 Nh5 Be3 Nf6 Qc2 Nfd7 Rbd1 a4 f4 Qa5 Bf2 Qb4 Bf1 a3 b3 Nf8 Re3 Qa5 g4 Na6 Nde2 Nb4 Qd2 Ne6 Bg2 Nc5 g5 Be6 Rg3 Rad8 Bd4 Bxd4 Nxd4 Bc8 Kh2 Ne6 Nde2 Qa7 Nb1 c5 Nbc3 Qb8 Nb5 Nc6 Rd3 d5 cxd5 Nb4 Rg3 Ng7
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 g3 Nf6 Bg2 g6 c4 Bg7 Nc3 d6 d5 c5 Nh3 Na6 Nf4 Nc7 h4 O-O h5 Qe8 a4 Ng4 hxg6 hxg6 Qd2 Bd7 b3 Bd4 e3 Bf6 Bb2 Qf7 Nd3 Qg7 Nd1 Ne5 Nxe5 Bxe5 Bxe5 Qxe5 Nc3 Kg7 f4 Qf6 Kf2 Rh8 e4 Rxh1 Rxh1 Rh8 Re1 fxe4 Nxe4 Qf5 Qe3 Kf8 b4 cxb4 Qxa7 Rh2 Kg1 Rh5 Qxb7 Ne8 a5 Bc8 Qxb4 Nf6 Nxd6 Qd3 Nb5 Rh7 Qc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 e3 Re8 Re1 e5 d5 Na5 e4 c6 Bg5 f6 Be3 Nac4 dxc6 bxc6 Bc1 Qxd1 Rxd1 Bf8 b3 Na3 Bb2 Bg4 Rac1 Red8 h3 Be6 Bf1 Kf7 Kg2 Ke8 Ba6 Rab8 Rxd8 Kxd8 Nd1 Kc7 Ne3 Bb4 Ng4 Bxg4 hxg4 g5 Rh1 Nd7 Rxh7 Kd6 Bxa3 Bxa3 Nd2 Bb4 Nc4 Kc7 Ne3 Rd8 Bc4 Kb8 Rf7 Bd2 Nf5 Bb4 Kf3 Ba3 Ng7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 g4 Nc6 g5 Nd7 Rg1 Be7 Be3 O-O Qd2 a6 O-O-O Nxd4 Bxd4 b5 f4 b4 Ne2 Qa5 Kb1 e5 Bf2 Nc5 Qe3 Be6 Nc1 exf4 Qxf4 Rac8 Bd4 Rfe8 Nb3 Qa4 Bf6 Bf8 Rg3 Qc6 Nxc5 dxc5 Be5 c4 Bd6 b3 c3 Rcd8 e5 Bxd6 exd6 Rxd6 Rc1 Qc5 Ka1 Red8 Re3 Rd1 Re1 Rxe1 Rxe1 Qa5 a3 Qd5 Be2 g6 h4 Qd2 Qf1 Bh3 Qg1 Re8
Result: 0-1
Processing game...
FEN: 
Moves: d4 d6 c4 e5 Nf3 e4 Ng5 Nf6 Nc3 Bf5 g4 Bxg4 Bg2 Be7 Ngxe4 Nxe4 Bxe4 c6 Qd3 Bh5 Qh3 Bg6 Bxg6 fxg6 Bf4 O-O e3 Na6 O-O-O Nc7 Kb1 a6 Ne4 g5 Bg3 Qe8 Ka1 Qg6 Qg2 Ne8 Rdg1 b5 c5 dxc5 Nxc5 Bxc5 dxc5 Rd8 h4 gxh4 Rxh4 Rd2 Rd4 Re2 Qh1 Qc2 Rb1 Qxc5 Qe4 Nf6 Qe6 Kh8 Be5 h6 Rh1 Rxf2 a3 Qc2 Rdh4 Qg6 Qxc6 Qf5 Rf4 Rxf4 exf4 Kh7 Rg1 Rf7 Qxa6 b4 Qc4 Nd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e6 e4 c5 e5 Ng8 Nf3 Nc6 d4 cxd4 Nxd4 Nxe5 Ndb5 a6 Nd6 Bxd6 Qxd6 f6 Be3 Ne7 Bb6 Nf5 Qc5 d6 Qa5 Qe7 O-O-O O-O f4 Nc6 Qa3 e5 g4 Nfd4 Nd5 Qf7 f5 g6 Rg1 gxf5 g5 Kh8 gxf6 Be6 Qxd6 Bxd5 cxd5 Qxf6 Qxf6 Rxf6 Kb1 Nd8 Bc5 Rc8 Be7 Rf7 Bd6 Nf3 Rg3 e4 Be2 Rf6 Bf4 Rg6 Bxf3 Rxg3 Bxe4 fxe4 hxg3 Kg7 Rd4 Nf7 Rxe4 Rd8 Re7 Rxd5 Rxb7 h5 Ra7 a5 a4
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nf3 d5 d4 Nf6 Bg5 dxc4 Nc3 a6 a4 Bb4 e3 c5 Bxc4 cxd4 exd4 Nc6 O-O O-O Rc1 Be7 Re1 h6 Bf4 Bd7 Ne5 Rc8 Nxd7 Qxd7 d5 exd5 Nxd5 Nxd5 Bxd5 Bg5 Bxg5 hxg5 h4 Rfd8 hxg5 Qf5 Rc5 g6 Qb3 Rb8 Qc4 Re8 Rxe8 Rxe8 Bxc6 Re1 Kh2 Qxf2 Bf3 Qg1 Kg3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 Nc3 c6 e4 b5 a4 e5 Nf3 exd4 Qxd4 Qxd4 Nxd4 b4 Nd1 Ba6 Bf4 Nf6 f3 Bc5 Nf5 O-O Rc1 c3 bxc3 g6 Ng3 Nfd7 cxb4 Bxb4 Bd2 c5 Nb2 Bxf1 Nxf1 f5 exf5 Rxf5 Ng3 Rd5 Bxb4 cxb4 Rc8 Kg7 Ke2 Nb6 Rc7 Rd7 Rhc1 a5 Ne4 Ra7 Rxa7 Rxa7 Rc5 N6d7 Rc8 Ra6 Kd3 Ne5 Kd4 Nbc6 Ke3 Ra7 Nd3 Nxd3 Kxd3 Ne5 Kd4 Nd7 Rc6 Rb7 Nd2 Nb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 g3 Nf6 Bg2 c6 d4 exd4 Qxd4 d5 Nf3 Be7 cxd5 cxd5 O-O Nc6 Qa4 O-O Be3 Be6 Nc3 Qd7 Rfd1 h6 Rac1 a6 Bb6 Rac8 Ne1 Rfe8 Nd3 Bd6 Nf4 Bxf4 Qxf4 Qe7 a3 Nb8 Bd4 Nbd7 e4 dxe4 Nxe4 Nxe4 Qxe4 Rxc1 Rxc1 Qd6 Qe3 b6 Rc6 Qb8 Qc3 f6 b4 Bf5 h3 Kh7 Be3 Qd8 Qd4 Ne5 Rxb6 Qc8 Kh2 Rd8 Qc5 Bxh3 Be4 Kg8 Qxc8 Bxc8 Bc5 f5 Bb1 Rd1 Ba2 Kh7 Rd6 Ng4 Kg2 Bb7 f3 Rc1 Be6 Rc2 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5 Qc2 Nxc3 bxc3 Be7 e3 Qc8 Bb2 c5 Bb5 Nc6 O-O O-O Bd3 Kh8 Qe2 Qc7 Rad1 Rad8 e4 Na5 Rfe1 Bf6 e5 Be7 Nd2 cxd4 cxd4 Bd5 Ne4 f5 exf6 Bxf6 Nxf6 Rxf6 Bc1 Rdf8 f3 Nc6 Be4 Bxe4 fxe4 e5 d5 Nd4 Qd3 Qf7 Bb2 Nf3 gxf3 Rxf3 Re3 Qg6 Kh1 Rf2 Rg1 Qh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 cxd5 exd5 Bg5 c6 Qc2 Be7 e3 Nbd7 Bd3 O-O Nge2 Re8 O-O Nf8 f3 Be6 Rae1 Rc8 Kh1 N6d7 Bxe7 Rxe7 Nf4 Rc7 Qf2 Nf6 e4 dxe4 fxe4 Rcd7 d5 cxd5 Bb5 Rc7 exd5 Bd7 Be2 Rc8 Qxa7 b6 Qa6 Ne4 d6 Nxd6 Nfd5 Re5 Qxb6 Nf5 Qxd8 Rxd8 Bd3 Rxe1 Rxe1 Ng6 a4 Nd4 a5 Kf8 Bxg6 hxg6 Rd1 Ne6 Nb6 Bc6 Rxd8 Nxd8 b4 Ne6 b5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 e5 Qd5 Bf2 Rd8 Qa4 b6 Qc2 Rf8 Kg1 Qc4 Qd2 Qe6 h3 Nc4 Qg5 h6 Qc1 Qf7 Bg3 g5 Qc2 Qd5 Bf2 b5 Ng3 Rf7 Re1 b4 Qg6 Kf8 Ne4 Rxf2 Kxf2 bxc3 Qf5 Kg8 Qc8 Kh7 Qxc5 Qf7 Kg1 c2 Ng3 Bf8 Nf5 Kg8 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 Bg5 exd5 Nxd5 Be6 O-O-O Bxd5 Rxd5 Qb6 Bxf6 Qxf6 e5 Qf5 Bd3 Qc8 Rd1 b5 Qh4 Nb4 Bxg6 fxg6 Rd7 Qe8 Re7 Bh6 Kb1 Rd8 Rd6 Qc6 a3 Rxd6 exd6 Qxd6 axb4 cxb4 Qe4 b3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 g6 e4 c5 c4 Bg7 d4 cxd4 Nxd4 Nc6 Be3 Nf6 Nc3 Ng4 Qxg4 Nxd4 Qd1 e5 Nb5 O-O Qd2 Qe7 O-O-O Nxb5 cxb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 Be7 Qd2 Nc6 g4 O-O O-O-O Nd7 h4 Nde5 Nxc6 bxc6 Be2 Rb8 g5 d5 Ba7 Rb7 Bd4 Nd7 exd5 cxd5 f4 Qa5 Qe3 Nb8 f5 Nc6 f6 Nxd4 Rxd4 Bc5 fxg7 Rd8 Rhd1 Qb4 b3 e5 Qxe5 Qxc3 Rxd5 Qxe5 Rxe5 Ba3 Kb1 Rxd1 Bxd1 Rd7 Bg4 Rd4 Bf3 Kxg7 h5 h6 gxh6 Kxh6 Ra5 Bb4 Re5 f6 Re8 Bf5 Rh8 Kg5 Rb8 Ba3 h6 Rd2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 c5 Bg2 Bg7 O-O cxd4 Nxd4 O-O c4 Nc6 Nc3 Nxd4 Qxd4 d6 Qd3 a6 Bf4 Bf5 e4 Be6 Rac1 Nd7 b3 Ne5 Qd2 b5 cxb5 axb5 Nxb5 Qa5 Nc3 Rfc8 Na4 Qxd2 Bxd2 Nd3 Rcd1 Bg4 Bg5 Bxd1 Rxd1 Nb2 Nxb2 Bxb2 Bxe7 Rxa2 Bxd6 Ra1 Rxa1 Bxa1 Bf1 Rc2 g4 Bd4 Bg3 f6 h4 h6 g5 hxg5 hxg5 fxg5 Bc4 Kg7 e5 Kh6 e6 Bc5 Kg2 g4 Bf4 Kh5 Be3 Bxe3 e7 Rxf2 Kg3 Rf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Nc3 d5 Bf4 Bg7 Qd2 Ne4 Nxe4 dxe4 Ne5 Be6 e3 Nd7 Nc4 O-O Be2 Bxc4 Bxc4 e5 dxe5 Nxe5 Bb3 a5 a4 Nd7 O-O-O Nc5 Qxd8 Rfxd8 Bg5 Re8 Rd5 Bf8 Rhd1 Nxb3 cxb3 Bd6 Bf6 Re6 Bc3 b6 g4 Bxh2 Rh1 Bd6 Rh3 f6 Kd1 Rf8 Ke2 Rf7 Rh1 f5 gxf5 Rxf5 Rxf5 gxf5 Rh5 f4 exf4 Bxf4 Rf5 Bd6 Bd4 h6 Ke3 Kh7 Rf7 Kg6 Rg7 Kf5 Rf7 Kg4 Rg7 Kh3 Rh7 Kh4 Rg7 h5 f4 exf3 Kxf3 c5 Bc3 Kh3 Rf7 Be7 Rf5 h4 Be5 Bd6 Rf6 Rxe5 Rxd6 Rf5 Ke2 Rf4 Rxb6 Kg3 Rg6 Rg4 Rf6 Re4 Kd3 Rf4 Rg6 Rg4 Rc6 h3 Rxc5 h2 Rc1 Kg2 Rc2 Kh3 Rc1 Rg3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Qa4 Qxa4 Nxa4 O-O Nf3 Ne4 Be5 Bd7 Nc3 Nxc3 bxc3 dxc4 Bxc4 Rc8 Bd4 e5 Bxe5 Rxc5 Bxg7 Kxg7 Bb3 Rxc3 O-O Na6 Ne5 Be8 Bd5 Rc7 Rab1 Nc5 e4 Rd8 Rfc1 Rdc8 g4 f6 Nf3 b6 Nd4 Bd7 f3 Nd3 Rxc7 Rxc7 Rd1 Nf4 Kf2 Kf8 Bb3 Ke7 Ne2 Nxe2 Kxe2 Rc3 h4 h6 e5 Bb5 Kf2 fxe5 Rd5 Rc5 Rxc5 bxc5 g5 hxg5 hxg5 Bd3 Bg8 Bf5 Bb3 Be6 Bc2 Bxa2 Bxg6 a5 Ke3 a4
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 g3 b6 Bg2 Bb7 O-O Be7 b3 d5 e3 Nbd7 Bb2 O-O Qe2 Ne4 Rfd1 Bf6 cxd5 exd5 d4 Qe7 Rac1 Rfd8 dxc5 Nxc3 Bxc3 bxc5 Qb2 Rac8 Qa3 Bxc3 Rxc3 Nf6 Rdc1 Ne4 R3c2 Rc6 Ne1 Ba6 Rd1 h5 Qc1 h4 Nd3 hxg3 hxg3 Bxd3 Rxd3 Rh6 Qd1 Qe5 a3 a5 f4 Qe6 g4 g5 Rc4 Rd6 Ra4 c4 bxc4 dxc4 Rxd6 Nxd6 Qd4 gxf4 Qxf4 Rg6 g5 Qf5 Qxf5 Nxf5 Rxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6 d4 c5 Nf3 cxd4 cxd4 Nc6 a3 h6 Bb2 Bd6 Bd3 O-O O-O e5 d5 Nb8 c5 Bxc5 Nxe5 Qf4 Rc1 b6 Rc4 Qg5 h4 Qd8 Qf3 Qd6 Rf4 f5 Nc4 Qd7 Re1 Bb7 Bxf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 Nc3 Nc6 e3 d5 cxd5 Nxd5 Bc4 Nxc3 bxc3 cxd4 exd4 Be7 O-O O-O Re1 b6 Bd3 Bb7 h4 Bxh4 Nxh4 Qxh4 Re3 g6 Rh3 Qf6 Bh6 Rfe8 Qg4 Rac8 Bg5 Qg7 Qh4 f5 Re1 Na5 Rhe3 Qf7 Bb5 Bc6 Bxc6 Nxc6 c4 Qd7 Bf6 Qf7 d5 exd5 Rxe8 Rxe8 Rxe8 Qxe8 cxd5 Qe1 Kh2 Ne5 d6 Nd7 Qc4 Kf8 Qc8 Qe8 Be7 Kf7 Qc4 Kg7 Qe6 g5 Bf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 g3 e6 Bg2 Nf6 c4 d5 Nd2 c6 Nh3 Bd6 Qc2 b6 Nf3 O-O O-O Qe7 Bf4 Bxf4 Nxf4 Ne4 Ne5 Bb7 cxd5 cxd5 Rfc1 Na6 Qa4 Nd6 e3 b5 Qa3 Rfe8 b3 Qd8 Bf1 Qb6 Nd7 b4 Nxb6 bxa3 Nxa8 g5 Bxa6 Bxa6 Nc7
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 g3 Bg7 Bg2 O-O d4 d6 Nf3 Nbd7 O-O e5 e4 exd4 Nxd4 Nc5 h3 Re8 Re1 a5 Rb1 a4 b4 axb3 axb3 c6 b4 Ne6 Nde2 Nf8 Be3 Be6 c5 d5 exd5 Nxd5 Nxd5 Bxd5 Bxd5 cxd5 Qd3 Ne6 Nd4 Qd7 Nb5 d4 Bd2 Ra2 Rbd1 Qc6 Nd6 Rea8 Qe4 Qxe4 Rxe4 R8a7 Kg2 Rc2 Be1 h5 h4 Rca2 Rd3 R7a3 Rxa3 Rxa3 Nxb7 Ra2 c6 Rc2 b5 d3 Nd8 Nc7 b6 Na6 Ra4 Bc3 Bxc3 Rxc3 Rd4 Kg7 Rd7
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e6 g3 Bb4 Bg2 O-O O-O d5 cxd5 exd5 Bg5 Bxc3 bxc3 Be6 Rb1 Na5 Rb5 a6 Bxf6 gxf6 Rb4 Rc8 e4 dxe4 Bxe4 f5 Bxf5 Bxf5 Nxf5 Qxd1 Rxd1 Rxc3 Rd7 Rf3 Rd5 b5 a4 Rb3 Rg4 Kh8 Nh6 Nc6 Rf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 d6 Bd3 Nc6 Ne2 Bxc3 Nxc3 O-O O-O e5 d5 Ne7 Qc2 Ng6 f4 exf4 exf4 c6 dxc6 bxc6 h3 d5 cxd5 cxd5 Qf2 Re8 Rd1 Bb7 Be3 Qa5 Bf1 Ne7 Bc5 Nf5 g4 Ne7 b4 Qd8 Bg2 Qb8 f5 h6 Re1 Nc6 Rxe8 Nxe8 Nxd5 Ne5 Rd1 f6 a4 Kh8 Nf4 Qc8 Ne6 Nd7 Bxb7 Qxb7 Qg2 Qxg2 Kxg2 Nb6 a5 Nc4 a6 Rb8 Bxa7 Rxb4 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 g3 b6 Bg2 Bb7 O-O Be7 b3 O-O Bb2 d6 e3 Nbd7 d4 cxd4 Nxd4 Qc8 Rc1 Bxg2 Kxg2 Qb7 Qf3 Qxf3 Kxf3 Rfc8 Rfd1 a6 Ke2 Nc5 f3 g6 g4 e5 Nc2 b5 cxb5 axb5 Nb4 e4 Nxb5 Rcb8 a4 Nxb3 Nc6 Nxc1 Rxc1 Kf8 Nxb8 Rxb8 Bxf6 Bxf6 fxe4 Be5 h3 Kg7 Kd3 h5 gxh5 g5 Rc7 Ra8 Ra7 Rh8 a5 Rxh5 a6 Rxh3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Nbd2 b6 e3 Bb7 Bd3 O-O O-O d5 a3 Be7 b4 Nbd7 cxd5 exd5 Rb1 a6 Ne5 Bd6 f4 Ne4 Nxe4 dxe4 Nxd7 Qxd7 Bc4 Be7 Bd2 Bd5 Rc1 c6 Qb3 Rfc8 Rc2 b5 Bxd5 cxd5 Rxc8 Rxc8 Ra1 Rc4 a4 g6 h3 Kg7 axb5 axb5 Ra6 Bh4 Kh2 Qc8 Ra7 Qc6 Qd1 Rc2 Qg4 Bf6 Qd1 Rb2 Be1 Qc4 Qg4 Qf1 f5 g5 Qg3 h5 Bc3 Rc2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 c5 g3 Bg7 Bg2 O-O O-O cxd4 Nxd4 Nc6 Nc3 Nxd4 Qxd4 d6 Qd3 a6 Bd2 Rb8 Rac1 Nd7 e4 b5 cxb5 axb5 b3 Nc5 Qe3 e6 Rfd1 b4 Ne2 Ba6 Be1 Qb6 Rc2 f5 exf5 exf5 Nf4 Rfe8 Qd2 Kh8 Nd5 Qd8 Nxb4 Re2 Qxe2 Bxe2 Rxe2 Qf8 Nd5 Re8 Rc2 g5 b4 Ne6 Bc3 f4 a4 Qf5 Bxg7 Kxg7 Rcd2 f3 Bf1 h5 a5 h4 Ne3 Qe4 Rxd6 hxg3 hxg3 Qxb4 a6 Rf8 a7 Qa5 Ra6 Qxa6 Bxa6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Nc3 d5 Bf4 Bg7 e3 O-O Be2 Bg4 h3 Bxf3 Bxf3 c6 O-O Nbd7 Qd2 Re8 Rfd1 Qb6 a4 a5 Ne2 e5 dxe5 Nxe5 Bxe5 Rxe5 Nd4 Ne4 Bxe4 Rxe4 c3 Rae8 Rab1 h5 b4 Qc7 Nf3 Qe7 Qd3 g5 Nd2 Re6 Nf1 g4 hxg4 hxg4 g3 Qg5 Kg2 Qh5 f4 gxf3 Kf2 Qh1 Rd2 Qg2 Ke1 Rxe3 Qxe3 Rxe3 Nxe3 Qg1 Nf1 Bxc3 Rb2 Qg2
Result: 0-1
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 Rb1 a5 a3 d6 d3 Nf6 b4 axb4 axb4 O-O b5 Nd4 e3 Ne6 Nge2 Re8 O-O Nc5 h3 h6 Bb2 Bd7 d4 exd4 Nxd4 Qc8 Kh2 c6 Qc2 Qc7 Ra1 Rxa1 Rxa1 h5 Kg1 Qc8 Kh2 Qc7 Kg1 Qc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 g3 e6 Nf3 d5 cxd5 exd5 d4 Nc6 Bg2 Be7 O-O O-O Be3 c4 Ne5 h6 h3 Bb4 Qa4 Bxc3 bxc3 Bf5 g4 Bh7 Bf4 Re8 Rad1 a6 Bg3 b5 Qa3 Qb6 Qc1 Ne4 Bxe4 Bxe4 Nd7 Qd8 Nc5 Bg6 Qd2 a5 f3 b4 e4 b3 axb3 cxb3 Nxb3 dxe4 fxe4 Bxe4 c4 Qb6 Nc5 Rad8 Qf2 Nxd4 Nxe4 Rxe4 Qxf7 Kh7 Kh2 Qc6 Rf2 Rd7 Qf8 Qxc4 Rdf1 Re2 Rxe2 Qxe2 Rf2 Qe4 Rf4 Qe2 Rf2 Qe6 Qa8 Qd5 Qf8 a4 h4 Rb7 Qa3 Qe4 Kh3 Rb3 Qf8 a3 g5 Qh1 Kg4 h5 Kf4 Ne6
Result: 0-1
Processing game...
FEN: 
Moves: c4 c6 e4 d5 cxd5 cxd5 exd5 Nf6 Bb5 Nbd7 Nc3 g6 d4 Bg7 d6 e6 Nf3 O-O Bf4 a6 Be2 b5 Ne5 Bb7 Bf3 Bxf3 Qxf3 b4 Ne4 Nxe4 Qxe4 Nf6 Qc6 Nd5 Bg3 Qa5 d7 b3 Ke2 bxa2 Rhc1 Rfd8 Qxa8 Rxa8 Rc8 Qd8 Rxd8 Rxd8 Rxa2 Bxe5 Bxe5 Rxd7 Rxa6 f6 Bd6 Rb7 Ba3 Kf7 g3 g5 h4 gxh4 gxh4 h5 Rc6 Kg6 f3 Rb3 Rc2 Kf5 Bd6 Nf4 Bxf4 Kxf4 Rc6 Rxb2 Kd3 Kf5 Rc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 Nf3 O-O Be2 e5 Be3 Ng4 Bg5 f6 Bh4 g5 Bg3 Nh6 h3 Nc6 d5 Nd4 Nxd4 exd4 Qxd4 f5 Qd2 f4 Bh2 Nf7 h4 h6 hxg5 hxg5 g3 f3 Bxf3 Ne5 Be2 g4 Bg1 c5 dxc6 bxc6 O-O-O Be6 Qxd6 Qg5 Kb1 Qg6 Ka1 Rab8 Rh5 Qxh5 Qxe6 Kh8 Qe7 Nf3 Bxf3 Rxf3 Na4 Re8 Qxa7 Qe5 Qb6 Qxe4 Qb4 Qc2 Rb1 Rd3 Qc5 Re2 Qh5 Kg8 Qxg4 Qxb1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 Nbd7 cxd5 exd5 Bd3 c5 Bf5 Qc7 Nh3 g6 Bxd7 Nxd7 O-O Rfe8 Nf2 Rac8 Qd2 h5 Rfe1 Qd6 Rad1 Qf8 dxc5 bxc5 e4 Nb6 exd5 Nxd5 Ne4 Re6 Qg5 Qg7 Rxd5 Bxd5 Qxd5 Qxb2 Rd1 c4 Qg5 c3 Rd8 Rxd8 Qxd8 Kh7 Bf6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 d5 cxd5 Nxd5 d4 Nxc3 bxc3 g6 e3 Bg7 Bd3 O-O O-O Qc7 Qe2 Rd8 Rd1 b6 Bb2 Nc6 Rac1 Bb7 e4 e6 Qe3 Rac8 h4 h6 Be2 Qe7 h5 g5 Ba3 Bf8 dxc5 Qf6 e5 Qf4 Qxf4 gxf4 Rxd8 Rxd8 Rd1 Rxd1 Bxd1 Nb8 Bc1 bxc5 Bxf4 Bd5 Bb3 Bxf3 gxf3 Nd7 Ba4 Nb6 Bb5 Nd5 Bd2 Bg7 f4 f6 c4 Nb6 Be3 Bf8 f5 Kf7 fxe6 Kxe6 exf6 Kxf6 Kg2 Ke5 Kf3 Nc8 Ba6 Nb6 Ke2 Nd7 f4 Kd6 Bb7 Nf6 Bf3 Ke6 Kd3 Kf5 a4 Bd6 a5 Bc7 Bxc5 Bxa5 Bxa7 Kxf4 Be2 Ng4 c5 Ne5 Kc2 Nc6 Bb6 Ke5 Kd3 Bxb6 cxb6 Kd6 Bf3 Nb8 b7 Kc7 Ke4 Kxb7 Kf5 Kc7 Kg6 Nd7 Bg2 Nf8 Kxh6 Ne6 Kh7 Kd6 h6 Ke7 Bh3 Nf4 Kg7 Nh5 Kg6 Nf4 Kf5 Kf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 Nc6 d5 Ne5 Nf3 Nxf3 Bxf3 h6 Be3 e6 O-O Re8 Rc1 exd5 cxd5 c5 dxc6 bxc6 e5 Rxe5 Bf4 Re6 Bxc6 Rb8 Qd2 g5 Bg3 Ba6 Rfe1 Rc8 Bf3 d5 Nxd5 Nxd5 Qxd5 Rxe1 Rxe1 Qxd5 Bxd5 Bxb2 h4 Bc4 Bxc4 Rxc4 hxg5 hxg5 Be5 Bxe5 Rxe5 Ra4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c5 d5 e6 Nc3 exd5 Nxd5 Nxd5 Qxd5 Nc6 e4 d6 Bc4 Be6 Qd3 Nb4 Bb5 Ke7 Qe2 Nxc2 Qxc2 Qa5 Bd2 Qxb5 Bc3 f6 a4 Qb3 Qe2 Qc4 Qe3 Kf7 Nd2 Qa6 f3 g6 Kf2 Bg7 Rhd1 Rhe8 Qf4 Rad8 Nf1 d5 exd5 Rxd5 Ne3 g5 Qc7 Re7 Qxe7 Kxe7 Nxd5 Kf7 Nc7 Qc4 Rd8 g4 Re1 Bf5 Ne8 Qxa4 Nxg7 gxf3 Nxf5 Qc2 Kxf3 Qxf5 Kg3 b5 Ree8 Qg5 Kf2 Qf4 Kg1 Qc1 Re1 Qc2 Rd6 f5 Rf6 Kg8 Re8 Kg7 Rxf5 Kg6 Rf6 Kh5 Re5 Kg4 h3 Kh4 Be1
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 g3 Bg7 Bg2 f5 Nc3 Nf6 Nf3 O-O O-O d6 d4 Qe8 d5 Na6 Be3 h6 Rc1 Bd7 a3 c5 dxc6 bxc6 b4 Nc7 c5 Ng4 Bd2 d5 h3 Nf6 Bf4 Ne6 Be5 Qf7 Nd4 Ne4 Bxg7 Qxg7 Nxe6 Bxe6 Qc2 Rfb8 Nxe4 fxe4 Qd2 a5 Rb1 axb4 axb4 g5 Rfc1 Ra3 e3 Rba8 b5 Ra2 Rc2 Ra1 Rxa1 Rxa1 Kh2 cxb5 c6 Qe5 Qb4 Qc7 Qxb5 Ra5 Qb7 Ra7 Qb4 h5 Qc5 h4 Rb2 Ra5 Rb5 Rxb5 Qxb5 Qd6 Qb6 Kf7 Qb5 hxg3 fxg3 Qa3 c7 Qc3 Qb6 Bd7 h4 gxh4 gxh4 Qe5 Kg1 Qg3 Qc5 Qe1 Bf1 Qg3 Bg2 e6 c8=Q Bxc8 Qxc8 Qe1 Kh2 Qxh4 Kg1 Qe1 Kh2 Qxe3 Qc7 Kf6 Qd8 Ke5 Qc7 Kd4 Qb6 Kd3 Qxe6 Qf4 Kg1 Qe3 Kh2 Qc5 Qe5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 cxd5 exd5 Bg5 c6 Qc2 Na6 e3 Nc7 Bd3 Ne6 Bh4 g6 Nf3 Be7 h3 O-O O-O Nh5 Bxe7 Qxe7 Rab1 Ng5 Qd1 f5 Nxg5 Qxg5 Qf3 Ng7 Ne2 Ne6 g3 Qh6 h4 a5 Nf4 Nxf4 Qxf4 Qxf4 gxf4 h5 Rfc1 Kf7 Kf1 Ke7 Ke2 Bd7 b3 Kd6 a3 Rfe8 Kd2 b6 b4 axb4 axb4 Ra2 Rc2 Rea8 Rxa2 Rxa2 Bc2 Be8 Kc3 Ra7 Bb3 Bd7 Kb2 Be8 Rc1 Bd7 Rc3 Be8 Bd1 Bd7 Be2 Be8 Bd1 Bd7 Bc2 Be8 f3 Bf7 Bb3 Be8 Rc1 Bf7 Re1 Be8 Rc1 Bf7 Rc3 Be6 Bc2 Bf7 Ra3 Rxa3 Kxa3 b5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 Be6 d3 O-O b4 Nd4 Bb2 Bb3 Qc1 c5 bxc5 Bxc5 Nxd4 Bxd4 Bxb7 Rb8 Bg2 Qd7 Ne4 Rbc8 Qb1 Bc2 Qa2 Bxd3 Bxd4 Bc4 Qb2 exd4 Rfd1 Rfd8 Bf3 f5 Nd2 Bf7 Qb4 Na4 Nb3 Rc4 Qe1 Nc3 Rd3 Qa4 Nxd4 Nxe2 Qxe2 Rcxd4 Rxd4 Qxd4 Rc1 h6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 Nbd7 cxd5 exd5 e3 Be7 Qc2 Nh5 Bxe7 Qxe7 O-O-O Nb6 h3 g6 g4 Ng7 Bd3 Be6 Ne2 O-O-O Nf4 Kb8 Be2 Ne8 Nd2 Nd6 h4 Bc8 Nb3 Ne4 Bf3 f5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be3 a6 Qd2 Be7 f3 Nc6 O-O-O O-O g4 Rb8 h4 Nxd4 Bxd4 Nd7 g5 b5 Kb1 Qc7 h5 b4 Ne2 Ne5 Ng1 f5 gxf6 Bxf6 h6 g6 b3 Bb7 Bh3 Qe7 Rh2 a5 Re2 Nf7 Bxf6 Qxf6 f4 Rbc8 Nf3 Qc3 Re3 Qxd2 Rxd2 Rc3 Rxc3 bxc3 Rd3 Bxe4 Rxc3 Bxf3 Bxe6 Bh5 Rc7 g5 fxg5 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 b6 f3 f4 Na4 axb4 axb4 g5 c5 Nf6 cxd6 cxd6 b5 Bd7 Nc4 Nc8 Ba3 Ne8 g4 fxg3 hxg3 g4 Bc1 gxf3 Bxf3 Nf6 Bg5 Ra7 Rf2 Rb7 Rb3 Ra7 Rb1 Rb7 Rb3 Ra7 Rb4 Kh8 Qf1 Bxb5 Rxb5 Rxa4 Bg2 h6 Bh4 Qe8 Bxf6 Rxf6 Rxf6 Qxb5 Re6 Kg8 Bh3 Rxc4 Rxh6 Bxh6 Be6 Kh8 Qf6
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 Be6 d3 O-O b4 Nd4 Bb2 Nxf3 Bxf3 c6 Ne4 Nd7 Qc2 Bd5 Nd2 Bxf3 Nxf3 Bd6 Nd2 Qe7 Ne4 Rae8 Nxd6 Qxd6 a4 f5 b5 c5 e3 Rf7 Rfe1 Nf8 Qc4 Rd8 Rad1 b6 Kg2 Ng6 f4 exf4 exf4 Qd5 Kf2 Rfd7 Rc1 h5 Qxd5 Rxd5 d4 cxd4 Rc7 R8d7 Rxd7 Rxd7 Rd1 Nf8 Rxd4 Rxd4 Bxd4 Ne6 Ke3 Kf7 Be5 Nc5 Kd4 Nxa4 Bb8 Ke6 Bxa7 g6 Kc4 h4 gxh4 Nb2 Kc3 Na4 Kb4 Nc5 Bxb6 Nd3 Kc4 Nxf4 Bf2 Nh3 Ba7 Kd7 Kd5 Nf4 Ke5 Nh5 Bc5 f4 Bf2 Ke7 Ke4 Ke6 Bd4 Kd6 Kf3 Kd5 b6
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O b3 d6 Bb2 Qe8 c4 Na6 d5 Bd7 Nc3 c6 Rc1 h6 e3 Rc8 Nd4 Qf7 Ba3 cxd5 Nxd5 Ne4 f3 Nec5 Nb5 Bxb5 cxb5 Nc7 Nxc7 Rxc7 Bxc5 dxc5 f4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 c5 Ngf3 Nf6 exd5 exd5 Bb5 Bd7 Bxd7 Nbxd7 O-O Be7 dxc5 Nxc5 Nd4 Qd7 Qf3 O-O N2b3 Nce4 Qf5 Rfd8 Re1 Bf8 c3 Qxf5 Nxf5 g6 Ne3 Re8 Rd1 Nc5 g4 h6 h4 Nxb3 axb3 Bc5 g5 hxg5 hxg5 Ne4 Ng4 Bb6 Kg2 Kg7 Bf4 Rad8 f3 Nc5 b4 Nb3 Ra3 Re2 Kg3 Rxb2 c4 Re8 c5 Bd8 Rxa7 Ree2 Rxd5 Be7 Rxb7
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 Bb4 Bg2 O-O O-O e4 Ng5 Bxc3 bxc3 Re8 f3 exf3 Nxf3 d5 d4 Ne4 Qc2 dxc4 Rb1 f5 g4 Qe7 gxf5 Nd6 Ng5 Qxe2 Bd5 Kh8 Qxe2 Rxe2 Bf4 Nd8 Bxd6 cxd6 Rbe1 Rxe1 Rxe1 Bd7 Re7 Bc6 f6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 d5 d4 Nbd7 cxd5 exd5 Bf4 Nb6 Qc2 g6 e3 Bf5 Bd3 Bxd3 Qxd3 c6 O-O Bg7 b4 O-O b5 Re8 bxc6 bxc6 Bg5 Qd6 Nd2 Nfd7 a4 Qe6 Bf4 a5 Bg3 Bf8 Ne2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 e3 Bb7 b3 d5 Nf3 Nbd7 Be2 dxc4 bxc4 c5 O-O Rc8 Bb2 cxd4 exd4 b5 c5 a6 a4 Nd5 Qd2 b4 a5 Qc7 Rfc1 Qf4 Bf1 Rfd8 Qxf4 Nxf4 Ra4 Nd5 Nd2 Rc7 Raa1 Nb8 Nc4 Bc6 Nd6 Ra7 f3 Ne7 Rc4 Nd5 Rcc1 Ne7 Rc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bd3 cxd4 cxd4 Be6 Rc1 Bxa2 Qa4 Be6 d5 Bd7 Qb4 e6 Rfd1 exd5 exd5 Re8 Bf2 b5 Nd4 Nc4 Nc6 Bxc6 dxc6 Nb2 Bxb5 Nxd1 c7 Qd5 Bxe8 Nxf2 c8=Q Rxc8 Rxc8 Nh3 gxh3 Qd1 Kg2 Qe2 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 d5 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Re8 Bf4 Bf5 Rad1 Ne4 Bd3 Bxc3 bxc3 b5 Qxb5 Nxc3 Qxa6 Bxd3 Qxd3 Ne2 Kh1 Nxf4 Qc4 Qd6 Rfe1 Rxe1 Nxe1 Rb8 a3 Rb2 f3 Qe5 Qe4 Qg5 g3 Qh5 h4 Ne2 Qe8 Kg7 d6 Nxg3 Kg1 Ne2 Kf1 Qf5 Qxe2 Qh3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 d4 d6 e4 O-O Be2 e5 Be3 Qe7 dxe5 dxe5 O-O c6 Nd2 Na6 Qa4 Ng4 Bxg4 Bxg4 f3 Be6 Rfd1 Rfc8 a3 Nc5 Qb4 Bf8 Na4 Nd3 Qc3 b5 Qxd3 bxa4 Qc2 Qb7 Rab1 Qa6 Qc3 Re8 b4 axb3 Rxb3 Rad8 Rdb1 h6 a4 Kh7 Rb7 Rd7 Rxd7 Bxd7 Nb3 Be6 Bc5 Qxc4 Qxc4 Bxc4 Bxf8 Bxb3 Rxb3 Rxf8 Rb7 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 f4 a6 Be2 Be7 a4 Qc7 O-O O-O Qe1 Nc6 Be3 e5 Nb3 Nb4 Qf2 Nxc2 Bb6 Qb8 Rad1 Be6 fxe5 Nd7 Nd5 Bxd5 Rxd5 Nb4 Rdd1 Nxe5 Nd4 g6 Ba5 Nbc6 Bc3 Nxd4 Rxd4 Qa7 Kh1 Rac8 Bd1 Qb6 a5 Qd8 Rb4 Nd3 Qd4 Rxc3 bxc3 Nxb4 cxb4 d5 Bb3 dxe4 Qxe4 Qd7 Qd5 Qxd5 Bxd5 Bxb4 Bxb7 Bxa5 Bxa6 h5 Bc4 Kg7 Ra1 Bb6 Rb1 Rd8 g3 Rd6 Be2 Bc7 Rc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bd3 cxd4 cxd4 Be6 Rc1 Bxa2 Qa4 Be6 d5 Bd7 Qb4 e6 Nc3 exd5 exd5 Re8 Bf2 Bf8 Qb2 Bg7 Qb4 Bf8 Qb2 Bg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 Qc2 b6 Nb3 axb4 axb4 fxe4 Nxe4 Nf6 Bd3 Nxe4 Bxe4 Nf5 Qd3 Qh4 g3 Qf6 f3 Bd7 Bd2 Nd4 Nxd4 exd4 Ra1 Bh3 Rxa8 Rxa8 Rd1 Bf5 Re1 h5 Kf2 Bxe4 Qxe4 Ra3 Qf4 Qxf4 gxf4 Bf6 Re8 Kg7 Rc8 Ra7 Ke2 Bh4 Kd3 Bf2 Bc1 Ra1 Rxc7 Kf6 Bd2 Ra3 Ke4 Ra2 Kd3 Ra3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 f4 a6 Be3 b5 Qf3 Bb7 Bd3 Nbd7 a3 Rc8 O-O Be7 Rae1 O-O Qh3 Nc5 Bf2 Nfd7 Kh1 Bf6 Rd1 Re8 Bg1 g6 Nde2 Bg7 f5 Qe7 Bd4 Nf6 Qg3 e5 fxg6 hxg6 Be3 Nfxe4 Bxe4 Nxe4 Nxe4 Bxe4 Nc3 Bb7 h4 Rc4 Bg5 Qe6 Rd2 Rg4 Qd3 f6
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Qa5 Bd2 Qd8 Bg5 Qa5 Bd2 Qd8 c4 Nxe4 cxb5 Be6 Nc3 Nxd2 Qxd2 Nd4 bxa6 Be7 Bd3 O-O O-O d5 Nc2 Nxc2 Qxc2 e4 Be2 f5 Nb5 Rxa6 Rfd1 Rb6 a4 Qb8 Qc7 Bf6 Qxb8 Rfxb8 Rab1 Kf7 b3 Rc6 b4 Rc2 Bf1 Bb2 Nd4 Bxd4 Rxd4 Kf6 b5 Ke5 Rbd1 Ra2 Rb4 Rb6 Rdb1 g5 R1b2 Ra3 R2b3 Rxb3 Rxb3 d4 Rb1 d3 a5 Rd6 a6 Bc8 h3 Bxa6 bxa6 Rxa6 g3 Rd6 Kg2 f4 gxf4 gxf4 Rb5 Kf6 Rb1 f3 Kh2 Ke5 h4 Kf4 Rd1 h5 Kh3 Rd8 Kh2 Rd6 Kh3 Rd4 Kh2 e3 fxe3 Kxe3 Bxd3 Rxd3 Ra1 Ke2 Kg3 Rd2 Rb1 Ra2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 g3 Bg7 Bg2 O-O d4 d6 Nf3 a6 O-O Nc6 d5 Na5 Nd2 c5 Qc2 Rb8 b3 b5 Rb1 Bd7 Bb2 bxc4 bxc4 Rb4 Nce4 Nxe4 Bxe4 Ba4 Qc1 Bxb2 Rxb2 Qb6 Rb1 Rxb1 Qxb1 Rb8 Qd3 Qd8 Qc3 Rb4 Bd3 Nb7 a3 Rb6 f4 Qf8 f5 Qg7 f6 exf6 Rxf6 Qh6 Kg2 Qe3 Rxf7 Qd4 Qxd4 cxd4 Rc7 Nc5 Ne4 Nxe4 Bxe4 Rb2 Bd3 Bd1 Kf1 Ra2 c5 dxc5 d6 Ra1 d7 Ba4 Kf2 Bxd7 Rxd7 Rxa3 Re7 Rc3 Bxa6 h6 h4 g5 h5 Kf8 Re6 Kf7 Rxh6 c4 Rc6 d3 Rxc4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 d6 Nc3 g6 e4 Bg7 Bd3 O-O Nge2 e6 O-O exd5 cxd5 a6 a4 Qc7 h3 Nbd7 f4 Rb8 Be3 Re8 Ng3 c4 Bc2 Nc5 Qf3 b5 axb5 axb5 e5 dxe5 fxe5 Rxe5 Bd4 b4 Bxe5 Qxe5 Rae1 Qd4 Kh1 Bxh3 Nce2 Qxb2 Qf4 Rc8 Bb1 Bd7 Nd4 Rf8 Nf3 c3 Qd6 Nb3 Ne5 Bb5 Rxf6 Bxf6 Qxf6 c2 Nf5 gxf5 Qg5 Kh8 Qh6 Kg8 Bxc2 Qxc2 Re3 f4 Qxf4 Qc1 Kh2 f6 Rg3 Kh8 Nf7 Rxf7 Qb8
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 g3 Nf6 Bg2 c6 d4 exd4 Qxd4 d5 cxd5 cxd5 Nf3 Nc6 Qa4 Be7 O-O O-O Be3 Ng4 Bd4 Nxd4 Nxd4 Qb6 Nc3 Qh6 h4 g5 Nxd5 Bd8 Rac1 gxh4 Rxc8 hxg3 Nf3 Nh2 Rfc1 Rxc8 Rxc8 Nxf3 exf3 gxf2 Kf1 Qd2 Nf6 Kg7 Ne8 Kh8 Qe4 Bh4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 d5 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Rd1 Re8 d6 h6 Bf4 Nd7 Rd2 Nb4 Qb3 Be6 Bc4 Nb6 Bxe6 Rxe6 Bg3 Qd7 a3 Nc6 Qb5 Rc8 Rad1 Bxc3 bxc3 Ne5 Qxd7 Nxf3 gxf3 Nxd7 Rb1 b6 a4 a6 Rdb2 Rb8 c4 a5 Kf1 f5 Bf4 g5 Bd2 Rbe8 Rxb6 Nxb6 Rxb6 Rd8 Bxa5 Rdxd6 Rb5 Kf7 Rxc5 Kg6 Kg2 Rc6 Rxc6 Rxc6 Bd8 Rxc4 a5 Ra4 Bb6 Ra3 Bd8 f4 Bb6 Kf5 h3 h5 Bd8 g4 hxg4 hxg4 fxg4 Kxg4 Bb6 f3 Kh2 Ra2 Bc5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Qa5 Bd2 Qd8 Bg5 Qa5 Bd2 Qd8 Bg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 d6 e4 c5 d5 Nbd7 Nh3 exd5 cxd5 a6 Bd3 h6 Bf4 Qe7 O-O b5 Rfe1 Ne5 Bf1 Bc8 Nf2 Ng6 Bd2 Bd7 b4 Rfc8 f4 cxb4 Qxb4 Rc2 Rec1 Rxc1 Rxc1 Ne8 Re1 Rc8 Bd3 Qf6 g3 h5 Be3 h4 Bd4 Qd8 Qd2 Nf6 Be2 Qe7 Bf3 Ne8 Re3 Qd8 Bd1 hxg3 hxg3 Nf8 Bb3 Ng6 e5 dxe5 fxe5 Qg5 Ne4 Qh5 e6 fxe6 Nc5 e5 d6 Kh7 Nxd7 exd4 Re2 Rc3 Qe1 Nh4 Nf8 Kh8 Ng6 Qxg6 Rxe8 Kh7 Bg8 Kh6 Qd2 Kh5 Re5 Nf5 Qe2 Kg5 d7 Qd6 Qe4 Rxg3 Kf2 g6 Rd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Bxc6 dxc6 d3 Nd7 b3 O-O Bb2 f6 Nbd2 Nc5 d4 exd4 Nxd4 Bd6 Qe2 Ne6 Qc4 Qe8 Nf5 b5 Nxd6 cxd6 Qd3 Qe7 Rfe1 Ra7 Nf1 Rd7 Ng3 d5 Nf5 Qc5 b4 Qa7 Qb3 Kh8 Rad1 dxe4 Nd6 Nf4 Qg3 Nd5 Rxe4 Qc7 Nxc8 Qxg3 hxg3 Rxc8 c4 bxc4 Rxc4 Kg8 Rc5 Kf7 a3 Ke6 Ra5 Ra8 Bc1 Rb7 Rc5 Kd7 Bd2 Rb5 Rc2 Re8 f3 Re6 Kf2 Rb8 g4 Rbe8 Bf4 Ke7 Bc1 h6 Rd4 Kd7 a4 Rb8 Rc5 Ke8 Bd2 Rb7 Ra5 Rb6 f4 Rd6 Kf3 Kd7 Ke4 Kc7 f5 Rd7 b5 axb5 axb5 Re7 Kd3 Rd7 bxc6 Nb4 Bxb4 Rxd4 Kxd4 Rxb4 Kc5 Rxg4 Ra7 Kc8 Ra2 h5 Rd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6 Bd3 b5 cxb5 cxd5 exd5 e4 Nxe4 Nxd5 Bg5 Qa5 Qd2 Qxd2 Bxd2 Bxb2 Rb1 Bg7 Ne2 Nd7 Nxd6 Nc5 Bc2 Be6 Ne4 Rac8 O-O Nxe4 Bxe4 f5 Bd3 Nb6 Nc1 Rfd8 Bg5 Rd7 Re1 Kf7 Be2 h6 Bh4 Nd5 Bd1 Bd4 Bf2 Bxf2 Kxf2 Nc3 Bb3 Bxb3 Rxb3 Nd1 Rxd1 Rxd1 Nd3 Rd2 Ke3 Rxg2 Ra3 Re8 Kd4 Re7 Ne5 Kf6 Nc6 Rd7 Kc4 Rc2 Kb4 Rxh2 Ra6 Kg5 a4 h5 Rxa7 Rxa7 Nxa7
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 d5 O-O O-O Ne5 c6 c4 Be6 cxd5 Bxd5 Bh3 c5 dxc5 Ne4 Nd3 Na6 Be3 Bc6 Qc2 Bd4 Bxd4 Qxd4 Bg2 Rac8 Nc3 Nxc3 bxc3 Qf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Bb5 Nd4 Bc4 e6 Nf3 Ne7 O-O Nec6 d3 g6 Nxd4 cxd4 Ne2 Bg7 Bd2 O-O b4 b6 b5 Ne7 Bb4 d6 a4 a5 Ba3 Bb7 Bb3 d5 f3 Qc7 Qe1 Rad8 Qh4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Be2 Be7 O-O O-O Kh1 Qc7 a4 Nc6 Be3 Re8 Bg1 Rb8 Bf3 Bd7 Nb3 b6 g4 Bc8 g5 Nd7 Bg2 Bf8 Qf3 Na5 Nd2 Bb7 Qh5 g6 Qh3 Rbc8 f5 Ne5 fxe6 fxe6 Rf4 Qe7 Raf1 Bg7 Rh4 Bh8 Bd4 Nac4 Nxc4 Rxc4 Be3 Rcc8 Bc1 b5 axb5 axb5 Rhf4 b4 Na2 Ba6 Rd1 b3 cxb3 Be2 Rd2 Qxg5 Qe3 Bh5 Rf8 Rxf8 Qxg5 Ng4 Nc3 Be5 h3 Bf4 Qe7 Rce8 Qd7 Ne3 Rf2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nc3 Nc6 e3 O-O O-O Re8 d5 Na5 Nd4 Bd7 b4 Nac4 a4 a5 b5 Qc8 Re1 Bh3 Bh1 Qg4 Nce2 Rad8 Qb3 Qd7 Nf4 Bg4 Ra2 Qd6 h3 Bc8 Bg2 e5 dxe6 Bxd4 exd4 Bxe6 Nxe6 Rxe6 Rxe6 Qxe6 Qc3 Qf6 Re2 Qxd4 Qxd4 Rxd4 Bh6 Nd6 Re7 Rxa4 Bxb7 Rc4 Bc6 a4 Be8 Re4 Bxf7 Nxf7 Rxe4 Nxh6 Re7 Nf5 Rxc7 Nd6 Ra7 Ndc8 Rc7 Nd6 Ra7 h6 Kf1 Kf8 Ra6 Ndc8 Ke2 Ke7 Kd3 Kd6 Kd4 g5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 Nbd7 Qd2 c6 Nf3 e5 O-O exd4 Nxd4 Nc5 Qf4 Qe7 Rad1 Re8 Bf3 a5 Rfe1 a4 h3 Qe5 Qc1 Qe7 Qf4 Ncd7 Qd2 Qf8 Nc2 h6 Be3 Nh7 Bd4 Ng5 Bg4 Ne5 Be2 Ne6 Be3 Nd7 Bf1 Ndc5 Nd4 Ng5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O Rb1 Na5 Bd3 cxd4 cxd4 b6 O-O e6 Qa4 Bb7 Rfd1 Rc8 Bd2 Nc6 Bc3 Qh4 Be1 Rfd8 f3 Qe7 Bb5 a6 Bxa6 Bxa6 Qxa6 Nxd4 Bf2 Ra8 Qd3 Ra3 Nxd4 Rxd3 Rxd3 Qd7 Rbd1 Qa4 f4 Qxa2 h4 Qc4 R3d2 b5 Nf3 Rxd2 Rxd2 Bf6 e5 Be7 Rd7 Qc1 Kh2 Qxf4 Kh3 Bf8 Rd8 Kg7 Be1 Bb4 Bg3 Qf5 Kh2 Bc3 Rb8 b4 Rb5 h6 Rc5 Qd3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 Nh4 Bc8 e3 e5 Bxc4 exd4 exd4 Be7 O-O Nd5 Re1 O-O Nf3 Be6 Qb3 Na6 Bxd5 cxd5 Bd2 Nb4 Ne2 a5 Nf4 Bg4 Re3 Rc8 Bxb4 Bxb4 Qxd5 Qc7 Nd3 Rfd8 Nxb4 axb4 Rae1 h6 Qb5 Bxf3 gxf3 Rxd4 Re8 Rxe8 Rxe8 Kh7 Qf5 g6 Qf6 Qc1 Kg2 Qg5 Qxg5 hxg5 Re4 Rxe4 fxe4 Kg7 e5 f6 exf6 Kxf6 Kf3 Ke5 Kg4 Kd4 Kxg5 Kc4 Kxg6 Kb3 f4 Kxb2 f5 b3 f6 Ka2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 d6 f3 d5 Bg5 Nbd7 e3 Re8 Nh3 h6 Bh4 c6 cxd5 exd5 Bf2 c5 Nf4 cxd4 Qxd4 Qa5 b4 Qb6 Qd2 a5 b5 Nc5 Rd1 Nb3 Qb2 a4 Be2 Qa5 Kf1 Bd7 Bh4 d4 Bxf6 gxf6 exd4 Bxb5 Kf2 Bxe2 Nxe2 Qg5 Rd3 Re7 Rhd1 Rae8 Ng3 f5 Kf1 Qh4 d5 Nc5 Rd4 Qxh2 d6 Re3 Kf2 Rxf3 Kxf3 Re3 Kxe3 Qxg3 Kd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 Qc2 Be7 e4 Nxc3 bxc3 O-O Bd3 c5 O-O Qc8 e5 h6 Qe2 Rd8 Rd1 Nd7 Be3 Qc6 Bb5 Qc7 Nd2 f5 a4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 Be3 h6 O-O Ng4 Bc1 Nc6 d5 Ne7 Nd2 f5 Bxg4 fxg4 b4 b6 Nb3 g5 a4 Ng6 a5 Bd7 c5 bxc5 bxc5 a6 Nd2 Nf4 Rb1 dxc5 Ba3 Rf7 Nc4 Qf6 Bxc5 Bf8 Bxf8 Raxf8 Rb4 h5 d6 Be6 Nd5 Bxd5 exd5 cxd6 Rb6 Rd7 Rxa6 Qg6 Re1 Nd3 Nxe5 Nxe5 Rxe5 Qf6 Re2 Re7 Rc6 Rxe2 Qxe2 Qa1 Qf1 Qxa5 Rxd6 Re8 Re6 Rxe6 dxe6 Qe5 Qc1 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 dxc4 Nc3 c6 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qb3 a5 Na2 Be7 Nh4 Be4 Nc3 Nb6 Be2 O-O Nxe4 Nxe4 Nf3 Nd5 Qxb7 Nb4 Ne5 Rc8 Bf3 Ng5 Bd2 Nxf3 gxf3 Bd6 Bxb4 Qg5 Kh1 Bxe5 dxe5 Qh5 Bxf8 Qxf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 cxd5 exd5 Bg5 c6 Qc2 Be7 e3 Nbd7 Bd3 Nh5 Bxe7 Qxe7 Nge2 g6 O-O-O Nb6 Ng3 Ng7 Kb1 Bd7 Rc1 O-O-O Na4 Nxa4 Qxa4 Kb8 Rc3 b6 Ba6 Ne6 Rhc1 Rhe8 Qb3 Qd6 Nf1 Ka8 Nd2 Nc7 Bf1 Ne6 g3 Rc8 Bg2 Rc7 h4 Rd8 Nf3 Bc8 Qa4 c5 Ng5 Nxg5 hxg5 Bb7 dxc5 bxc5 Qf4 Qxf4 gxf4 d4 Rxc5 Rxc5 Bxb7 Kxb7 Rxc5 dxe3 fxe3 Re8 Re5 Rxe5 fxe5 Kc6 Kc2 Kd5 b4 Kxe5 a4 f6 gxf6 Kxf6 b5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 Qa5 Nf3 Nc6 Rb1 a6 Rc1 cxd4 cxd4 Qxd2 Kxd2 f5 Bd3 Rf8 exf5 Bxf5 Rc5 Rd8 Bxa6 Nxd4 Bxd4 bxa6 Kc3 Bxd4 Nxd4 Bd7 f3 Rf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 e3 Bb4 Qc2 O-O d3 Re8 Bd2 Bxc3 Bxc3 d5 cxd5 Nxd5 Be2 Bf5 Rd1 a5 O-O Qe7 a3 a4 Be1 Bg6 Qc4 Rad8 Nd2 Nd4 exd4 Nf4 Bf3 Rxd4 Qb5 c6 Bxc6 bxc6 Qxc6 Qd8 Nf3 Rd6 Qb5 Rd5 Qb4 e4 Bc3 Ne2
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 d4 Nf6 g3 Bg7 Bg2 O-O Nf3 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 e3 e5 d5 Na5 e4 c6 Bg5 f6 Be3 cxd5 Bxb6 Qxb6 Nxd5 Qd8 b4 Nc6 Rc1 Bg4 Qb3 Kh8 b5 Bxf3 bxc6 bxc6 Rxc6 Bxg2 Kxg2 f5 Qc4 fxe4 Qxe4 Rb8 Rfc1 Qa5 R1c2 Rbd8 R6c5 Qa6 h4 Rd7 Nc7 Qf6 Rc6 Qf7 Ne6 Re8 Ng5 Qf8 Rc7 Qe7 Rxd7 Qxd7 h5 h6 Nf3 gxh5 Nh4 Qg4 Qd5 e4 Rc7 Rf8 Qd6 Kh7 Qd7 Qg5 Rxa7 e3 Qd3 Kg8 Qxe3 Qxe3 fxe3 Re8 Nf5 Bf8 Kf3 Re5 e4 Rb5 Kf4 Rb2 e5 Rf2 Ke4 Re2 Kd5 Rb2 e6 Rd2 Ke5 h4 gxh4 Bb4 e7 Re2 Kf6 Bc3 Kg6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O Bg4 Be3 Nc6 Qd2 e5 d5 Ne7 Rad1 Bd7 Ne1 b5 a3 a5 b4 axb4 axb4 Ra3 f3 Qb8 Nd3 c6 dxc6 Bxc6 Nc1 Rd8 Nb3 d5 Bc5 Qc7 Na5 Be8 Nb1 Ra2 Nc3 Ra3 Nb1 Ra2 Nc3 Ra3 Bxe7 Qxe7 Nxd5 Nxd5 exd5 e4 d6 Qa7 Kh1 Bc3 Qd5 Bxb4 Nc6 Bxc6 Qxc6 Qc5 Bxb5 Qxc6 Bxc6 exf3 Bxf3 Rxd6 Rxd6 Bxd6 Bd5 Ra7 c4 Kg7 g3 f5 Rf2 Kf6 Re2 Re7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 cxd5 exd5 Bf4 Nf6 e3 Bf5 Nge2 O-O Ng3 Be6 Bd3 c6 Qc2 Bd6 Nge2 Re8 h3 g6 O-O Nh5 Bxd6 Qxd6 Rab1 a5 a3 Nd7 Na4 Ng7 b4 axb4 axb4 b5 Nc5 Bf5 Rfc1 Nf6 Ra1 Bxd3 Nxd3 Rxa1 Rxa1 Nd7 Ra7 Ne6 Qa2 Ng5 Qa6 Re7 Qc8 Kg7 Qc7 Qf6 Ne5 Ne6 Qxd7 Rxd7 Rxd7 Nd8 Nf4 Kh6 Rxd8 g5 Nfd3 Kg7 Rd7 Qe6 Rxf7 Kg8 Nc5 Qc8 Ncd7
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 e6 e3 Nc6 d4 d5 cxd5 Nxd5 Bc4 cxd4 exd4 Be7 O-O O-O Re1 Nf6 a3 a6 Ba2 Qd6 Bg5 b5 Bxf6 Bxf6 Ne4 Qd8 d5 exd5 Bxd5 Bd7 Rc1 Rc8 Nd6 Rc7 Qb3 Ne5 Nxe5 Bxe5 Nxf7 Bxh2 Kxh2 Qh4 Kg1 Rxc1 Rxc1 Be8 Ne5 Kh8 Qe3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 f3 Nbd7 c4 e5 d5 Be7 Be3 c6 Nc3 a6 Bd3 O-O Nge2 Nh5 Qd2 g6 O-O Ng7 Kh1 f5 exf5 gxf5 f4 e4 Bc2 c5 g4 fxg4 Nxe4 Nf6 N2g3 Nxe4 Bxe4 Bf6 Qc2 h5 Bg6 h4 Nh5 b5 f5 bxc4 Qxc4 Bb7 Rad1 Qe7 Bh6 Qe5 Rfe1 Qxb2 Rb1 Qa3 Nxf6 Rxf6 Rxb7 Qf3 Kg1 Nxf5 Bh7 Kh8 Bxf5 Rxf5 Qe2 c4 Qxf3 gxf3 Kf2 Rg8 Rg1 Rxg1 Kxg1 Rxd5 Kf2 Rd3 Rb6 d5 Rxa6 c3 Rc6 d4 a4 Kh7 Bf4 Rd1 Kxf3 Ra1 Rc4 Rd1 Ke4 d3 Rxc3 d2 Kd3 Ra1 Bxd2 Rxa4 Rc4 Ra3 Bc3 Kg6 Rxh4 Kf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bb5 cxd4 cxd4 Bd7 Rb1 Rc8 d5 Bxb5 Rxb5 a6 Bb6 Qd6 Rb1 Nc4 Bd4 b5 Bxg7 Kxg7 a4 Qc5 Qd4 Qxd4 Nxd4 Nd2 axb5 axb5 Nc6 e6 Rxb5 Nxf1 Kxf1 exd5 exd5 Rc7 Ke2 Ra8 Rb2 Rd7 Rd2 Re8 Kf2 h5 h4 Kf6 Kg3 Rd6 Kf4 Rc8 Ke3 g5 hxg5 Kxg5 Ke4 Rcxc6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 c5 dxc5 Qa5 Rc1 dxc4 Bxc4 O-O Nf3 Qxc5 Nb5 Be6 Bxe6 Qxb5 Bb3 Nc6 Qe2 Qb4 Qd2 Qxd2 Nxd2 Nh5 Bd5 Nxf4 exf4 Nb4 Be4 Bxb2 Rc7 Rac8 Rxc8 Rxc8 Ke2 Nxa2 Bxb7 Rc2 Be4 Nc3 Kd3 Rxd2 Kxd2 Nxe4 Kc2 Bd4 Re1 Nf6 Rxe7 Bxf2 h3 h5 Rb7 Nd5 Rb3 Bb6 g3 Kg7 Kd3 h4 Ke4 hxg3 Rxg3 Nxf4 h4 Nh5 Rg5 Bd8 Rg2 Bxh4 Ra2 f5 Kf3 g5 Rxa7 Kg6 Ra6 Nf6 Ra5 g4 Kf4 Bg5 Kg3 f4 Kg2 Ne4 Ra3 Kf5 Rb3 Bf6 Rd3 Be5 Ra3 Ng5 Kf2 f3 Ra4 Ne4 Ke3 Bf4 Kd3 Nf2 Kd4 g3 Ra5 Kg4 Ra8 g2 Rg8 Bg5
Result: 0-1
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 e6 d4 cxd4 Nxd4 Bb4 g3 O-O Bg2 d5 cxd5 Nxd5 Bd2 Nxc3 bxc3 Ba5 O-O Bb6 Bc1 Qe7 a4 Na6 a5 Bc5 Nb3 Bd6 Be3 Nc5 Nxc5 Bxc5 Bxc5 Qxc5 Qd4 Qxd4 cxd4 Rb8 Rfb1 Rd8 Rb4 Kf8 Rab1 Rd7 e3 Rc7 g4 Ke7 Be4 h6 Kg2 g6 h4 Kd6 Kg3 f5 Bf3 e5 dxe5 Kxe5 h5 fxg4 Bg2 g5 Rd4 Be6 Rb5 Kf6 Rd6 Re7 Bd5 Kf7 Rxe6 Rxe6 Kxg4 Kf6 Bxe6 Kxe6 f4 gxf4 Kxf4 b6 Ke4 Kf6 axb6 Rxb6 Ra5 Rb4 Kf3 Rb7 Kf4 Rb4 e4 Rb7 Ra6 Kg7 Kf5 Rf7 Ke5 Rf1 Rxa7 Kg8 Ke6 Rh1 e5 Rxh5 Kf6 Rh1 Ra8 Kh7 e6 Rf1 Ke7 Rb1 Rd8 Kg7 Ke8 h5 e7 Kg6 Rd6 Kg5 Kf7 Rf1 Kg7 Re1 Rg6 Kf4 Kf7 Rxe7 Kxe7 h4 Ra6 h3 Kf6 h2 Ra1 Kg3 Kf5 Kg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 g6 e4 d6 d4 Bg7 Be2 O-O Nf3 Bg4 Be3 Nbd7 h3 Bxf3 Bxf3 c6 O-O a6 Rc1 e5 d5 c5 a3 Ne8 b4 Qc7 Qb3 b6 bxc5 dxc5 h4 Nd6 h5 f5 hxg6 hxg6 g3 b5 cxb5 f4 Bd2 c4 Qb4 a5 Qb1 Nc5 b6 Qf7 Nb5 Nxb5 Qxb5 Nb3 Bg4 Qe7 Rcd1 fxg3 Be6 Kh8 Kg2 Qh4 fxg3 Qxe4 Kg1 Nd4 Qd7 Nf3 Rxf3 Qxf3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 g3 b6 Bg2 Bb7 O-O g6 Nc3 Bg7 d4 cxd4 Qxd4 d6 b3 Nbd7 Bb2 O-O Rfd1 Rc8 Nd5 a6 Rac1 b5 Ne1 Nc5 Qd2 Nfe4 Qc2 Bxb2 Qxb2 bxc4 Rxc4 Bxd5 Rxd5 Nf6 Rd1 Qb6 Qc2 a5 Nd3 Nxd3 exd3 Rxc4 dxc4 Rc8 Qe2 e6 h3 Qc5 Qd2 Rd8 Qb2 Nd7 Qd4 Qxd4 Rxd4 Nc5 Bc6 f5 Kf1 Kf7 Ke2 Ke7 Bb5 h5 h4 Rg8 Rd1 g5 hxg5 Rxg5 Rh1 e5 Bc6 Ne6 Bd5 Nd4 Kd3 Kf6 Rb1 Rg4 b4 axb4 Rxb4 h4 gxh4 Rxh4 Rb6 Rh3 Kd2 Ra3 Rxd6 Ke7 Rb6 Rxa2 Ke1 e4 c5 Nf3 Kf1 Ra1 Kg2 Rg1 Kh3 Ng5 Kh2 Nf3 Kh3 Ng5 Kh4 Nf3 Kh5 Rg2 c6 Rxf2 c7 Rh2 Kg6 Rg2 Kxf5 Rg5 Kxe4 Kd7 Rc6 Nd2 Kf4
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 a5 d3 O-O Be3 Be6 Na4 Nd5 Bc5 Bd6 Rc1 h6 Nd2 Rc8 Ne4 b6 Nxd6 cxd6 Bxb6 Nxb6 Rxc6 Rb8 Nxb6 Rxb6 Qc2 Qb8 Rxb6 Qxb6 Rb1 Bb3 Qd2 Rb8 Rc1 Be6 Rc2 d5 h4 a4 Bf3 Qd4 e3 Qb6 d4 e4 Be2 Qb3 Kg2 g6 Bd1 Kg7 Qc1 Qb6 Rd2 Bd7 Qc5 Qxc5 dxc5 Bc6 Kf1 Kf6 Ke1 Ke5 Rc2 Rb5 Kd2 d4 exd4 Kxd4 Be2 Rxc5 Rxc5 Kxc5 Ke3 f5 g4 Kd5 gxf5 gxf5 Kf4 Ke6 Bc4 Kf6 b4 axb3 Bxb3 Bb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 a5 d3 O-O Be3 Be6 Na4 Nd5 Bc5 Bd6 Rc1 h6 d4 exd4 Nxd4 Nxd4 Qxd4 b6 Bxd6 Qxd6 Rfd1 Rfd8 e4 Ne7 Qxd6 Rxd6 Rxd6 cxd6 f4 b5 Nc3 b4 Nb5 bxa3 bxa3 Ra6 Rc7 Rb6 Bf1 Kf8 Kf2 g6 Ke3 d5 e5 g5 Bd3 gxf4 gxf4 d4 Kxd4 Nd5 Ra7 Nxf4 Bf1 Rc6 Nc3 Rc8 Rxa5 Rd8 Ke3 Ng6 Rc5 Ne7 Bd3 Nf5 Bxf5 Bxf5 Nd5 Ra8 Rc3 Be6 Nf6 Ra4 Ne4 Kg7 Nc5 Rh4 a4 Rxh2 a5 Ra2 a6 Bc8 Kf4 Bxa6 Rg3 Kf8 Nd7 Ke7 Nf6 Ra4 Ke3 Ke6 Rg7 Kxe5 Rxf7 Ra3 Kf2 Ke6 Rf8 Ke7 Nh7 Bd3 Rh8 Bxh7 Rxh7 Kf6 Rxh6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 g3 g6 Bg2 Bg7 O-O O-O Nc3 Nc6 Rb1 d5 cxd5 Nxd5 Nxd5 Qxd5 b3 Qd6 d3 e5 a3 Bd7 b4 cxb4 axb4 Nxb4 Ba3 a5 Qb3 b5 Ng5 Rac8 Bxb4 axb4 Qxb4 Qxb4 Rxb4 Rc2 Re1 h6 Ne4 Rfc8 Nd6 R8c5 Rbb1 Ra2 Bf3 h5 Rec1 Rxc1 Rxc1 Ra6 Ne4 Bf8 h4 b4 Rc7 Be6 Rb7 Ra1 Kg2 Rc1 Ng5 Ba2 Ra7 Rc2 Rb7 Bc5 Ne4 Bf8 Ng5 Kg7 Ra7 Bc5 Rb7 Bf8 Ra7 Bb3 Nxf7 Bc5 Rb7 Bxf7 Bd5 Rxe2 Rxf7 Kh6 Kf1 Rd2 Bc4 Rb2 f4 exf4 gxf4 b3 Rb7 Rf2 Ke1 b2 Bg8 g5 hxg5 Kg6 Bh7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 c3 d5 Nbd2 Nbd7 e3 O-O b4 c6 Be2 Re8 O-O e5 a4 h6 Bh4 a5 b5 c5 dxe5 Nxe5 Nxe5 Rxe5 Bxf6 Bxf6 Rc1 b6 Bg4 Bb7 Bf3 Qe7 c4 Rd8 Qc2 d4 Bxb7 Qxb7 exd4 Rxd4 Rce1 Rxe1 Rxe1 Qd7 Nf1 h5 g3 h4 Ne3 Qe6 Qe2 Re4 gxh4 Bc3 Rd1 Rxh4 Qf3 Bd4 Ng2 Rh3 Qd5 Qf6 Ne3 Rf3 Kh1 Rxf2 Ng4 Qf3 Qxf3 Rxf3 Re1 Ra3 Re8 Kg7 Rb8 Rxa4 Rxb6 Rxc4 Ra6 Rb4
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 d4 e6 Nc3 d5 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nbd7 Nge2 Re8 Qc2 Nf8 O-O-O Be6 Kb1 Ng4 Bxe7 Qxe7 Nf4 Nf6 f3 c5 g4 cxd4 exd4 Qd6 Qd2 a6 Nce2 Re7 Ng3 Ng6 Ng2 Nd7 Rhg1 Ree8 Rdf1 Ngf8 Ne3 Kh8 Nh5 g6 f4 gxh5 f5 h4 fxe6 fxe6 g5 Re7 Ng4 Rg7 Nh6 Qb6 g6 hxg6 Nf7 Kg8 Qh6 Rh7 Rxg6 Nxg6 Qxg6 Rg7 Qh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6 Qd2 cxd5 cxd5 Nbd7 Nge2 a6 Nc1 Nh5 Bd3 f5 N1e2 Ndf6 exf5 gxf5 Ng3 e4 Nxh5 Nxh5 fxe4 f4 Bf2 Bg4 h3 Bd7 O-O-O Be5 Kb1 Qf6 Be2 Ng3 Bxg3 fxg3 Bf3 Rac8 Ne2 Qg6 Rc1 Rxc1 Qxc1 Rc8 Qe3 Qf6 Qd2 Rc5 Nc1 Bf4 Qb4 Bb5 Nb3 Bd3 Ka1 Rc2 Rb1 Be5 Nc1 Bxb2 Qxb2 Qxb2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 Qd2 b5 f3 Nbd7 g4 h6 O-O-O Bb7 Bd3 Ne5 Rhe1 b4 Na4 d5 exd5 Nxd5 Bf2 Bd6 Bf5 O-O Nxe6 fxe6 Bxe6 Kh8 Bxd5 Nxf3 Bxf3 Bxf3 Bb6 Bxd1 Bxd8 Raxd8 Qxd1 Bf4 Kb1 Rxd1 Rxd1 Bxh2 c3 bxc3 Nxc3 Rf4 Rh1 Rf2 Re1 Kh7 Re2 Rxe2 Nxe2 Kg6 Nd4 Bd6 Nf3 h5 gxh5 Kxh5 Kc2 g5 Kd3 g4 Ke4 g3 Ne1 Kg4 Ng2 Kh3 Kf3 a5 b3 Be5 Ne1 Bc7 Ng2 Bd6 Ne1 Kh2 Ng2 Be5 Ne3 Kh3 Ng2 Bd6 Ne1 Be7 Nd3 Kh2 Nf4 Bg5 Ne2 g2 Ke4 Be7 Kf3 Bb4 Ke4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 cxd5 exd5 Bf4 c6 Qc2 g6 e3 Bf5 Qd2 Nf6 f3 c5 Bh6 cxd4 exd4 a6 g4 Be6 Nge2 Nbd7 Bg2 Nb6 b3 Rc8 O-O Rc6 h3 Nfd7 Nd1 Rg8 Nf2 f5 Rae1 g5 gxf5 Bf7 Ng4 Bh5 Ng3
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 d6 d4 c5 Be3 cxd4 Bxd4 Nf6 Nd5 Nbd7 Nf3 O-O Bg2 e5 Bc3 Nxd5 cxd5 Nc5 O-O Bd7 Nd2 Rc8 Rc1 f5 Nc4 Bb5 Na3 Be8 Qd2 Rf7 Bb4 Rfc7 Rc2 Na6 Bc3 e4 Bxg7 Kxg7 Rfc1 Bf7 g4 fxg4 Bxe4 Qf6 Nb5 Rxc2 Rxc2 Rxc2 Qxc2 Nc5 b4 Nxe4 Qxe4 h5 a3 g5 Nd4 a6 Ne6 Kh6 Kg2 Bg6 Qe3 Qf5 Qd4 Qe4 Kg1 Qxd4 Nxd4 Be4 f3 gxf3 exf3 Bb1 Kf2 Kg6 b5 Ba2 bxa6 bxa6 Nc6 Kf5 Nb4 Bc4 Ke3 a5 Nc6 Bxd5 Nd4 Ke5 Ne2 Bf7 Ng1 Bc4 Nh3 Kf5 Nf2 d5
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 Nf3 c5 c4 dxc4 Nc3 cxd4 Qxd4 Qxd4 Nxd4 Bd7 Ndb5 Na6 e4 Nf6 f3 Bxb5 Nxb5 e5 Be3 Bb4 Kf2 Ke7 Bxc4 Rhc8 Rac1 Bc5 Rhd1 Bxe3 Kxe3 Ne8 Bb3 Rxc1 Rxc1 f6 a3 Nd6 Bd5 Nxb5 Bxb7 Nbc7 Bxa8 Nxa8 Rc8 Nb6 Rg8 Kf7 Rh8 Nc5 Rb8 Ke7 b4 Nc4 Ke2 Nd7 Rg8 g5 a4 a5 bxa5 Nxa5 Ra8 Nc6 a5 Kd6 g3 h5 h4 gxh4 gxh4 Nc5 a6 Kc7 a7 Nb7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O d4 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 e3 Re8 d5 Na5 Nd4 Bd7 Qe2 c6 dxc6 Nxc6 Nxc6 Bxc6 Bxc6 bxc6 Rd1 Qc8 Bd2 Qe6 Be1 a5 Rac1 a4 Qc2 Rec8 b3 axb3 Qxb3 Qxb3 axb3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O d4 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 d5 Na5 e4 c6 Bg5 Bg4 h3 Bxf3 Qxf3 cxd5 Nxd5 Nxd5 Rad1 Qc7 exd5 Bxb2 Rfe1 Rfe8 Rxe7 Rxe7 d6 Re1 Rxe1 Qxd6 Re7 Rf8 Qd5 Qb6 Be3 Qf6 Rc7 b6 Rxa7 Kg7 Rd7 Re8 Kh2 h5 Bxb6 Re5 Bd4 Bxd4 Qxd4 Re2 Qxf6 Kxf6 Ra7 Rxa2 Bd5 Rxf2 Kg1 Rf5 Rxa5 Kg5 h4 Kh6 Rb5 f6 Kg2 g5 hxg5 Kxg5 Kh3 Re5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3 Qxc3 Ne4 Qc2 c5 dxc5 Nc6 cxd5 exd5 Nf3 Qa5 Bd2 Qxc5 Qxc5 Nxc5 Be3 Ne4 Nd4 O-O Rd1 Be6 f3 Nd6 Nxe6 fxe6 Bc5 Rad8 e4 Rfe8 exd5 exd5 Kf2 b6 Be3 Re5 Bd3 Rf8 Rc1 Na5 Bd4 Re7 Rhe1 Rd7 Bc3 Nb3 Rcd1 Nc5 Re5 Nf7 Re3 Nd6 Bb1 Nc4 Re2 Rfd8 Ba2 b5 b3 d4 bxc4 dxc3 Rxd7 Rxd7 cxb5 Kf8 Rc2 Rb7 Bc4 Na4 Ke3 Re7 Kd4 Rd7 Ke5 Rb7 Kd6 Rb6 Kd5 Rh6 Bb3 Nb6 Kc5 Rxh2 a4 Ke7 Kc6 Nd7 Rxc3 Rxg2 f4 h5 Re3 Kd8 Rd3 Rg6 Kb7 h4 Kxa7 Rg3 Rxg3 hxg3 Bd5 Kc7 Bg2 Kd6 b6 g6 a5 Kc5 b7 Kb4 a6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Bg4 Qb3 Nh6 Bxh6 Be6 Bxe6 fxe6 Bf4 Bc5 Qxe6 Qe7 Qxe7 Kxe7 Nbd2 Nc6 O-O-O Nb4 Kb1 Nd3 Bg3 Kd7 Nb3 Bb6 Rxd3 c5 Rhd1 c4 Ne5 Ke7 Nxc4 Rac8 Nxb6 axb6 Nxd4 Kf7 Nf5 g6 Rd7 Ke8 Ng7 Kf8 Ne6 Ke8 Rxb7 Ra8 Bh4 g5 Bxg5 Rg8 Re7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bxc3 bxc3 h6 Bh4 Bb7 Nd2 d6 f3 Nbd7 e4 e5 Be2 Qe7 Qa4 O-O O-O g5 Bf2 Kh7 Bd3 Rg8 Rfe1 c5 Nf1 Nf8 Ne3 Nh5 Bf1 Ng7 Rad1 Ng6 Nf5 Nxf5 exf5 Nf8 dxe5 dxe5 Bd3 f6 Be4 Rg7 g4 Bxe4 fxe4 Rd8 Rd5 Rxd5 cxd5 Qd7 Qa6 Qb7 Qa3 Nd7 c4 Nb8 Rb1 Na6 Qa4 Nb4 Rb3 Qd7 Qxd7 Rxd7 a3 Na6 Kf1 Kg8 h4 Kf8 Ke2 Rg7 Rh3 Nc7 hxg5 Ne8 Rxh6 Rxg5 Kf3 Kg7 Rh1 Kf7 Be3 Rg7 Rh6 Nd6 g5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 Nc6 e4 Nb6 d5 Nb8 Nc3 Bg4 a4 e6 a5 N6d7 h3 Bxf3 Qxf3 Qf6 Qxf6 Nxf6 Bf4 Bd6 e5 Nxd5 Nxd5 exd5 exd6 c6 Bd3 O-O Kd2 g6 Rhe1 Nd7 Re7 Nc5 Ree1 Nb3 Kc3 Nxa5 b4 d4 Kxd4 c5 bxc5 Nc6 Kd5 Ne7 Rxe7 Rfe8 Bb5 Rec8 d7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d6 Nf3 O-O O-O Nbd7 Qc2 c6 Nc3 e5 Rd1 Qc7 e4 a5 h3 Re8 Be3 exd4 Nxd4 Nc5 Re1 h5 Rad1 a4 f4 a3 b4 Ne6 Nb3 Bh6 Qf2 Ng7 Bb6 Qb8 Bd4 Nd7 Be3 Nf8 Nd4 Nfe6 Nf3 Bd7 Qd2 Rd8 g4 hxg4 hxg4 Ne8 f5 Bxe3 Qxe3 Nf8 e5 gxf5 exd6 Nxd6 c5 Ne4 gxf5 Nxc3 Qg5 Kh8 Qh6 Nh7 Ng5 Bxf5 Rxd8 Qxd8 Nxf7 Kg8 Nxd8 Rxd8 Re3 Nxa2 Rg3 Kf7 Rg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 Nc6 e4 Ndb4 d5 Nxd5 exd5 Nb4 Nc3 e6 Bc4 Be7 O-O O-O Bf4 Bd6 Bxd6 Qxd6 dxe6 Bxe6 Qxd6 cxd6 Bxe6 fxe6 Rad1 d5 Rfe1 Rf6 Nd4 Raf8 Nxe6 Re8 Nd4 Rxe1 Rxe1 Nd3 Re2 Rf4 Nf3 d4 Nd5 Nc1 Nxf4 Nxe2 Nxe2 d3 Nc3 h6 Kf1 g5 Ke1 g4 Ne5 h5 Nxd3
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 g6 e4 Bg7 d4 cxd4 Nxd4 Nc6 Be3 Nf6 Nc3 Ng4 Qxg4 Nxd4 Qd1 Ne6 Rc1 d6 b4 O-O Qd2 b6 Bd3 Bb7 O-O Rc8 Nd5 Re8 a4 Nd4 Rb1 e6 Nc3 d5 exd5 exd5 Nxd5 Bxd5 cxd5 Qxd5 Rfd1 Red8 h3 Rc3 Bf1 Qc6 Qa2 Rc2 Qa3 Rc3 Qa2 Rc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d5 exd5 Qxd5 Nc3 Qd6 d4 c6 Nf3 Nf6 h3 Bf5 Bd3 Bxd3 Qxd3 Nbd7 O-O e6 Ne4 Nxe4 Qxe4 Nf6 Qe2 Be7 c4 O-O Bd2 Rfe8 Bc3 Rac8 Rad1 b6 Ne5 Qc7 Rfe1 Rcd8 Qf3 c5 dxc5 Bxc5 Qc6 Qxc6 Nxc6 Rxd1 Rxd1 a5 Kf1 Ne4 Be1 Nf6 Ke2 Rc8 Ne5 Kf8 b3 Ke7 g4 h6 f3 g5 Nd3 Nd7 h4 f6 Bg3 e5 hxg5 hxg5 Rh1 Ke6 Rh7 Be7 Bf2 Nf8 Rh8 Rd8 c5 bxc5 Nxc5 Bxc5 Bxc5 Kf7 Bxf8 Rxf8 Rh7 Ke6 Ra7 Rc8 Rxa5 Rc2
Result: 1-0
Processing game...
FEN: 
Moves: e4 g6 d4 Bg7 Nc3 d6 Be3 Nd7 f4 e5 dxe5 dxe5 f5 Ngf6 g4 c6 Qf3 b5 Bd3 Qa5 Ne2 b4 Nd1 Nc5 O-O gxf5 gxf5 Nxd3 cxd3 Ba6 Nf2 Rd8 Rfd1 Bf8 Kh1 Rg8 Nc1 Be7 Nb3 Qc7 Rac1 h5 Bc5 Ng4 Bxe7 Nxf2 Qxf2 Bxd3 Nc5 Qb6 Nxd3 Rxd3 Qxb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Qa4 Bd7 Qa3 O-O Nf3 b6 Bg5 Re8 Bc4 Bc6 Qb3 Rf8 Nd2 Qd7 O-O Bb7 Rad1 Nc6 Qa3 e6 Rfe1 h6 Bh4 g5 Bg3 Ne7 Nf3 Ng6 d5 e5 h4 g4 Nh2 h5 Nf1 a6 Ne3 Ne7 Bb3 b5 c4 Rfe8 c5 c6 d6 Ng6 Nf5 Bf6 Qc1 Nf4 Bxf4 exf4 Qxf4 Bg7 Qg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 d6 Nc3 a6 a4 e6 e4 e5 Nf3 h6 h3 Be7 Bd3 Bd7 a5 Qc8 Be3 Bd8 Qb1 b6 axb6 Bxb6 b4 Qc7 bxc5 Bxc5 Bxc5 dxc5 O-O O-O Qa2 Nh5 Qa5 Qc8 Rfe1 f6 Bf1 Nf4 Nb1 g5 Nbd2 Bxh3 gxh3 Nxh3 Bxh3 Qxh3 Re3 Qc8 Nf1 h5 Ng3 Qg4 Qc7 Nd7 Kf1 Rfd8 Nf5 Kh7 Qb7 Rg8 Qc6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 g3 d5 cxd5 Nf6 Bg2 Nxd5 Nc3 Nb6 Nf3 Nc6 O-O Be7 a3 O-O d3 Bg4 b4 Nd5 Bb2 Nxc3 Bxc3 Bf6 Nd2 Qd7 Rc1 Bh3 Ne4 Bxg2 Nxf6 gxf6 Kxg2 a5 b5 Qd5 e4 Qxb5 Qg4 Kh8 Qh4 Kg7 Bd2 Rfd8 Bh6 Kg6 g4 Rxd3 Qh5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 Bg5 Nbd7 e3 g6 cxd5 Bg7 dxe6 fxe6 Nf3 O-O Bd3 c5 O-O cxd4 exd4 Qe8 Re1 Rf7 Nb5 Qf8 Nc7 Nb6 Nxa8 Nxa8 Rc1 Nb6 Ne5 Nfd5 Nxf7 Qxf7 Bc4 Nb4 Qb3 Bxd4 Qxb4 Bxf2 Kh1 Bxe1 Qxe1 e5 Qd2 Kg7 Bh6 Kf6 Rf1 Bf5 g4 Qxc4 Qd6 Kf7 Qf8 Ke6 gxf5 gxf5 Qxf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 e6 e4 Nb6 Nc3 Bb4 Bd3 O-O O-O h6 Qe2 Nc6 Be3 Ne7 Rad1 Bxc3 bxc3 Na4 Qc2 Bd7 c4 b6 e5 Ng6 Be4 Rb8 d5 Qe7 dxe6 Bxe6 Bd5 Nc5 Bxe6 fxe6 Qxg6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7 Nf3 Nbc6 h4 Qc7 h5 f5 h6 g6 a4 Qa5 Qd2 Bd7 Bd3 O-O O-O cxd4 cxd4 Qxd2 Bxd2 f4 a5 Nf5 Rfb1 Rab8 Bxf5 Rxf5 c3 Rff8 Nh2 g5 Ng4 Be8 Nf6 Kh8 g3 fxg3 fxg3 Ne7 Bxg5 Ng6 Rb2 Bc6 Rf1 Rbc8 Rf3 Rc7 Bc1 Rfc8 Nh5 Ba4 Bd2 Bd1 Kg2 Bxf3 Kxf3 Rf8 Nf6 Ne7 Ke2 Ng6 g4 Nh4 Kd3 Ng6 Rb3 Ne7 g5 Nc6 Rb5 a6 Rc5 Nxe5 Ke2 Rxc5 dxc5 Nc4 Bf4 Rc8 Nxd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 Ng5 e6 f4 Nc6 c3 Qe7 O-O h6 Ne4 Bd7 Nbd2 Na5 Bc2 Rc8 Nf3 Bc6 Qe1 Bxe4 Qxe4 f5 exf6 Bxf6 b3 Nd7 Ba3 Qg7 Rad1 O-O c4 c5 d5 Rf7 Rfe1 Re7 Bc1 b5 Bd2 b4 Rc1 Nb6 dxe6 Rxe6 Qxe6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Qc2 O-O Bd3 Nbd7 Nge2 h6 Bh4 Re8 O-O Qc7 Bg3 Bd6 Nb5 Qd8 Nxd6 Re7 Nf5 Re8 Bh4 Nf8 Neg3 Bxf5 Nxf5 Ne6 f4 Rc8 Rf3 Kh8 Rh3 Rg8 Rf1 Qf8 Bxf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O cxd4 cxd4 Bg4 f3 Na5 Bd3 Be6 d5 Bxa1 Qxa1 f6 Rb1 b6 Bh6 Re8 Nf4 Bf7 e5 Bxd5 Rd1 e6 Bb5 f5 Bxe8 Qxe8 Rxd5 exd5 e6 d4 Qxd4 Qe7 Nd5 Qc5 Qxc5 bxc5 Nf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 Nbd7 cxd5 cxd5 Bf4 e6 e3 Ne4 Bd3 f5 O-O Be7 Nb5 O-O Nc7 Rb8 Nxe6 Qb6 Nxf8 Bxf8 Bxb8 Nxb8 Ne5 Qxb2 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 Nc6 Nf3 Bg4 d5 Ne5 Nc3 c6 Bf4 Ng6 Be3 cxd5 exd5 Ne5 Qd4 Nxf3 gxf3 Bxf3 Bxc4 Qd6 Nb5 Qf6 Qc5 Qb6 Qa3 e6 Nc7 Qxc7 Bb5 Qc6 Bxc6 bxc6 Bc5 Bxc5 Qxf3 Bb4 Ke2 cxd5 Qg4 Be7 Rhc1 Kf8 Rc7 Bd6 Rb7 Nf6 Qa4 a5 Rc1 h6 Rc6 Ne8 b4 Bxh2 bxa5 Kg8 Qb4 Bd6 Rxd6 Nxd6 Rb8 Rxb8 Qxb8 Kh7 Qxd6 Rc8 a4 Rc4 Qd7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 e6 d4 d5 exd5 exd5 Nf3 Bd6 Be3 c4 b3 cxb3 axb3 Ne7 Na3 Nbc6 Nb5 Bb8 Bd3 Bf5 c4 O-O Ra4 Qd7 Nc3 Bc7 Bxf5 Qxf5 Nh4 Qd7 O-O Rad8 Re1 Rfe8 c5 Ba5 Qd3 a6 h3 Bxc3 Qxc3 Nf5 Nxf5 Qxf5 Ra2 Re6 Rae2 Rde8 Qd2 f6 Qc3 h5 b4 R8e7 Kh1 g5 Kg1 g4 h4 Re4 Qb2 Na7 Qd2 R4e6 Qc1 Nb5 Qd2 Na3 Qd1 Kf7 Qb3 Nc4 Kh2 Re4 g3 Qf3 b5 a5 c6 f5 cxb7 Rxb7 Kg1 f4 gxf4 g3 Qd1 Rbe7 b6 gxf2 Rxf2 Qxd1 Rxd1 Rxe3 Rg2 Nxb6 Rg5 a4 Rxh5 a3 Rd2 Re2
Result: 0-1
Processing game...
FEN: 
Moves: d4 d6 e4 g6 c4 e5 Nf3 exd4 Nxd4 Bg7 Nc3 Nc6 Be3 Nge7 h4 h6 Be2 f5 exf5 Nxf5 Nxf5 Bxf5 Qd2 Qd7 O-O O-O-O b4 Nxb4 Nb5 Nc2 Bf3 d5 Bxd5 Nxa1 Nxa7 Kb8 Qb4 Qxd5 cxd5 Nc2 Qa5 Nxe3 fxe3 Rhe8 Nb5 Rxd5 Qxc7 Ka8 Qa5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 Nc6 O-O e5 d5 Ne7 a4 a5 Ne1 Nd7 Be3 f5 f3 Nc5 Nd3 b6 b4 Nxd3 Qxd3 axb4 Nb5 Kh8 Qb3 Ng8 Qxb4 Nf6 exf5 gxf5 Bg5 h6 Bh4 Bd7 Ra3 Qb8 Bf2 Rg8 Rb1 Nh5 Kh1 Qd8 a5 Bxb5 Qxb5 bxa5 g3 f4 g4 e4 fxe4 Nf6 Bh4 Qe7 Bf3 Qe5 Re1 Nh7 Qd7 Bf6 Bxf6 Nxf6 Qf5 Qxf5 exf5 Nxg4 Bxg4 Rxg4 Rh3 Kg7 Re7 Kf6 Re6 Kxf5 Rhxh6 Rf8 Rh5 Rg5 Rxg5 Kxg5 c5 a4 cxd6 cxd6 Rxd6 Ra8 Re6 a3 Re1 Kf5 h4 f3 d6 a2 Ra1 Kf4 Kh2 f2 d7 Kf3 Rc1 a1=Q
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 O-O O-O Nb6 Nc3 Nc6 e3 Re8 d5 Na5 Nd4 Bd7 b3 c5 dxc6 Nxc6 Nxc6 Bxc6 Bxc6 Qxd1 Rxd1 bxc6 Bd2 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 g3 d6 Bg2 g6 d4 exd4 Qxd4 Nf6 Nc3 Bg7 Qe3 Qe7 Qxe7 Kxe7 b3 a5 Bb2 c6 Na4 Be6 Nf3 Nbd7 Nd4 Rhc8 O-O Rab8 Rac1 Ne8 Rfd1 c5 Nb5 Bxb2 Nxb2 Nb6 Nc3 Na8 Nba4 Nec7 Rd2 Bd7 Ne4 Ne8 Nac3 Bc6 Rcd1 Rd8 g4 Nac7 a4 Ne6 e3 h6 f4 Rd7 h4 Rbd8 Ng3 N6g7 Nd5 Kf8 Kf2 Ne6 g5 hxg5 hxg5 N6c7 Bh3 f5 gxf6 Rh7 Nxc7 Nxc7 Bg2 Ne8 Bxc6 bxc6 Kg2 Rb8 Rd3 d5 cxd5 cxd5 Rxd5 Rxb3 Ne4 Rxe3 Re5 Rh5 Re6 Re2 Kf3 Rhh2 f7 Rhf2 Nxf2 Rxe6 fxe8=Q Kxe8 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6 Be2 exd5 cxd5 Bg4 O-O Nbd7 h3 Bxf3 Bxf3 Re8 Re1 Rc8 Be3 b5 Nxb5 Nxe4 Bxe4 Rxe4 Nxd6 Rxe3 Rxe3 Bd4 Qf3 Rb8 Kh2 Nf6 Nc4 Bxe3 Qxe3 Qxd5 Qe5 Re8 Qxd5 Nxd5 f5 Kg7 Rd1 Nf6 fxg6 hxg6 Rd2 Re1 Nd6 Ne4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d5 cxd5 Nxd5 O-O Nc6 d4 Nb6 e3 e5 d5 Na5 e4 c6 Bg5 f6 Be3 cxd5 exd5 Bg4 Bc5 Rf7 b3 f5 Bb4 Nac4 bxc4 e4 Rc1 Qd7 Nb1 exf3 Bxf3 Bxf3 Qxf3 f4 g4 Qa4 Qb3 Qd7 f3 h5 h3 Re8 Qd3 hxg4 hxg4 Re3 Qxg6 Rf6 Qg5 Qf7 Bc3 Rg6 Qd8 Re8 Qh4 Bxc3 Rxc3 Qg7 Re1 Qd4 Kg2 Rxe1 Qxe1 Nxc4 Qe8 Kg7 Qe7 Kh6 Qh4 Kg7 Qe7 Kh6 Qf8 Kh7 Qf7 Kh6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Bg5 dxc4 Nc3 Bb4 e4 c5 e5 cxd4 Qa4 Nc6 O-O-O Bd7 Ne4 Be7 exf6 gxf6 Bh4 Rc8 Kb1 Na5 Qc2 e5 Nxd4 exd4 Rxd4 Qb6 Nxf6 Bxf6 Qe4 Kf8 Bxf6 Qxf6 Rxd7 Re8 Qd4 Re1 Kc2 Qxd4 Rxd4 Ke7 g3 Rd8 Bg2 Re2 Kd1 Rxd4 Kxe2 Kd6 h4 b5 h5 h6 Rd1 Rxd1 Kxd1 f5 Kd2 b4 Ke3 b3 axb3 Nxb3 Bh3 Ke6 f3 a5 g4 Ke5 f4 Kf6 Bf1 fxg4 Bxc4 Nc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3 Qxc3 Ne4 Qc2 e5 e3 exd4 cxd5 Qxd5 Nf3 Qc6 Nxd4 Qxc2 Nxc2 Nc6 b4 Be6 Bb2 O-O Rc1 Nd6 b5 Na5 Nd4 Bc4 Bc3 Bxf1 Rxf1 Nac4 Bb4 Rfc8 Ke2 a6 Bxd6 Nxd6 a4 axb5 axb5 Ra5 Rb1 Ra2 Kf3 Re8 Ra1 Rb2 Rfb1 Rxb1 Rxb1 Re5 g4 h5 h3 hxg4 hxg4 g6 Ke2 Rc5 Kd3 b6 f3 f5 gxf5 gxf5 e4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 Nc3 c5 e4 d6 d4 cxd4 Nxd4 Bb7 Qe2 Nbd7 g3 Rc8 Bg2 a6 O-O Qc7 b3 e6 Nd5 Qb8 Rd1 g6 Bg5 Bg7 Bxf6 Nxf6 Nxb6 Rd8 e5 Bxg2 exf6 Bxf6 Nxe6 fxe6 Qxe6 Be7 c5 Bb7 Re1 Qc7 c6 Bxc6 Rac1 Rd7 Nxd7 Qxd7 Qc4 Bb7 Qc7 Rf8 Qb8 Kf7 Rc7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 dxc4 Nc3 c6 a4 Bf5 Nh4 Bc8 e3 e5 Bxc4 exd4 exd4 Be7 O-O O-O Re1 Na6 Bg5 Nc7 Qb3 Nfd5 Bxe7 Nxe7 Nf3 Ncd5 Re5 Ng6 Re4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 Nd2 Qe8 O-O Nh7 a3 Bd7 b3 f5 exf5 gxf5 Bh5 Qc8 Be7 Re8 Bxe8 Qxe8 Bh4 e4 Qc2 Qh5 Bg3 Rf8 Bf4 Qg4 g3 Ng5 Kh1 Nf3 Rac1 Nc5 Nxf3 Qxf3 Kg1 Nd3 Qd2 Bd4 Rc2 Kh7 h3 Rg8 Kh2 Qh5 Nd1 Ne5 f3 Nd3 Ne3 Nxf4 gxf4 Bb6 Qf2 Qg6 Re2 Bc5 fxe4 fxe4 f5 Qh5 Rd2 Rg5 Qf4 Qe8 Ng4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Be7 f4 O-O O-O a6 Kh1 Qc7 Qe1 b5 e5 dxe5 fxe5 Qxe5 Nb3 Qc7 Bf4 e5 Bf3 Nc6 Bg5 Be6 Qg3 Kh8 Bxc6 Qxc6 Qxe5 Bd6 Qd4 Ng4 Bf4 Be7 Qe4 Qxe4 Nxe4 Bd5 Nc3 Bc4 Rfe1 Bh4 g3 Bf6 h3 Bxc3 bxc3 Nf6 Kh2 Rfe8 Be5 Nd7 Bd4 h5 a3 Rac8 Nd2 Bd5 Nf1 f6 Rxe8 Rxe8 Ne3 Bc6 Kg1 Ne5 Kf2 Nf3 h4 Nd2 Nf1 Nc4 Ne3 Nd6 Rg1 Bd7 Ke2 Bf5 Kf3 Bc8 Ke2 a5 Bc5 Ne4 Bd4 Ba6 Kf3 Nd2 Kf2 Bb7 Ke2 Nc4 Ra1 a4 Kf2 Nd6 Rg1 Bc8 Kf3 Kg8 Kf2 Kf7 Kf3 Bb7 Ke2 Re6 Kf2 g5 hxg5 fxg5 Ke2 Kg6 Kf2 Ba6 Rd1 Ne4 Kg2 Bb7 Kh2 Nxc3 Bxc3 Rxe3 Rd3 Re2 Kg1 Rg2 Kf1 Rxc2 Bb4 Kf5 Rc3 Rb2 Rc5 Kg4 Ke1 Rb3 Rxb5 Be4 Ra5 Rxg3 Be7 Bf5 Rxa4 Kh3 Bf6 Rg1 Kf2 Rg2 Ke1 h4 Ra8 Kg4 Ra4 Kf3 Bxg5 Re2 Kf1 Bh3 Kg1 Rg2 Kh1 Rxg5 Rxh4 Kg3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 O-O O-O Nb6 Nc3 Nc6 e3 Re8 d5 Na5 Nd4 Bd7 e4 c5 dxc6 Nxc6 Nxc6 Bxc6 Qb3 Qd3 Be3 Rac8 Rfd1 Qa6 Bd4 Qa5 Bxg7 Kxg7 Rac1 Red8 Rxd8 Rxd8 Rd1 Rxd1 Qxd1 Qc5 Nd5 e5 Nxb6 Qxb6 Qd2 Qc5 h4 h6 b4 Qc4 Qb2 f6 Qd2 g5 hxg5 hxg5 a3 Kf7 Qd6 Qd4 Qc7 Qd7 Qb8 Qd1 Kh2 Qh5 Kg1 Qd1 Kh2 Qd4 b5 Bxb5 Qxb7 Bd7 Kg1 Ke8 Bf1 Qc5 Qb8 Ke7 Qg8 Be6 Qh7 Bf7 Kg2 Qc8 Be2 Qg8 Qxg8 Bxg8 f4 gxf4 gxf4 Kd6 Kf3 Kc5 fxe5 fxe5 Ke3 Bc4 Bd3 Bb3 Be2 Ba4 Bd3 Bc6 Be2 Bb5 Bd1 Bf1 Bb3 a5 Ba2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 f3 g6 Be3 Bg7 Qd2 O-O c4 e5 d5 Nh5 Nc3 Qh4 Bf2 Qf4 Be3 Qh4 g3 Nxg3 Qf2 Nxf1 Qxh4 Nxe3 Ke2 Nxc4 Rc1 Na6 Nd1 Nb6 Ne3 Bd7 Nh3 f6 Nf2 Nc8 Rc3 Ne7 Rhc1 Rac8 Rb3 Rb8 Nd3 Rf7 Qe1 Nc8 Qa5 Nb6 Rxc7 f5 Rc2 fxe4 fxe4 Rbf8 Rxb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Be7 O-O O-O f4 a6 Kh1 Qc7 a4 Re8 Be3 Nc6 Qd2 Bd7 Nb3 b6 Bf3 Rab8 g4 Bc8 g5 Nd7 Qf2 Bf8 h4 Bb7 h5 Na5 Rad1 Nc4 Bc1 Rbc8 Bg2 Nc5 Nd4 d5 e5 Ne4 Nxe4 dxe4 b3 Na5 Be3 g6 Rd2 Red8 hxg6 hxg6 Qh4 Nc6 c3 Ne7 c4 Rxd4 Bxd4 Nf5 Qh3 Kg7 Qc3 a5 c5 Nxd4 Rxd4 Bxc5 Rc4 Rh8 Bh3 e3 Kh2 Qc6 Qc2 Ba6 Kg3 Bxc4 Qxc4 Qd7 Bg4 Qd2 Be2 Rd8 Rc1 Rd4 Qc2 Qb4 Qc3 Rxf4 Qxb4 Rxb4 Bc4 Be7 Kf3 Bxg5 Ra1 Kf8 Ra2 Ke7 Rg2 e2 Rxe2 b5 Bxb5 Rxb3 Kg4 Be3 Rc2 Bd4 Kf4 Rh3 Rc8 Rh4 Kg3 Rh8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 cxd5 exd5 Bg5 c6 e3 Be7 Bd3 Nbd7 Nge2 Nh5 Bxe7 Qxe7 g4 Nhf6 Ng3 h6 h3 Nb6 Qd2 Bd7 b3 g6 a4 a5 f3 h5 g5 Qd6 Nge2 Ng8 e4 Ne7 Qf4 Qxf4 Nxf4 O-O Nce2 h4 Ng2 Kg7 Kd2 dxe4 fxe4 Rad8 Raf1 Rh8 Ne3 Rh5 Rfg1 Bc8 Kc3 Rhh8 Nf4 Rd6 Bc2 Na8 Nc4 Rdd8 Nxa5 Nc7 Rf1 b5 Rf3 Rhf8 Rhf1 bxa4 bxa4 f6 Nxg6 Nxg6 gxf6 Kh6 Nxc6 Rd6 d5 Rxc6 dxc6 Ne6 e5 Nxe5 Re3 Ng6 f7 Ngf4 Bb3 Kg7 Re4 Rxf7 Bxe6 Nxe6 Rxf7 Kxf7 Rxh4 Ke7 Rh8 Ba6 h4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 g5 Bg3 Ne4 e3 c6 Bd3 Nxg3 hxg3 Be6 a3 Bf8 f4 gxf4 gxf4 Bg4 Bf5 Bxf5 Qxf5 Nd7 e4 Qf6 Qxf6 Nxf6 e5 Nd7 Nge2 Rg8 Kf2 Nb6 b3 Kd7 Ng3 Rc8 Kf3 c5 dxc5 Rxc5 Nce2 Rc6 Rhc1 Rxc1 Rxc1 Bxa3 Ra1 Bc5 Nf5 Ra8 Nxh6 Ke6 g4 a5 Rc1 Nd7 Nc3 Ra6 Re1 d4 f5 Ke7 Nd5 Kf8 e6 fxe6 fxe6 Nb8 Nf5 Nc6 g5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nc3 Nc6 e3 O-O O-O Re8 d5 Na5 Nd4 Bd7 Nb3 Nxb3 axb3 c6 dxc6 Bxc6 e4 Qxd1 Rxd1 Rad8 Be3 Nc8 f3 f5 exf5 gxf5 Kf2 a6 Bf1 e6 Bc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c6 O-O Re8 Qb3 Nd7 Rad1 a6 a4 Nf8 a5 bxa5 bxa5 Rb8 Na4 Ne6 Qc3 Rc8 Rb1 Rc7 Qd2 Qc8 Rfc1 Be7 Qb2 Bd6 Bf1 Qd7 Qb6 Ra8 Ne1 Qe8 Nd3 Re7 Qb2 Ra7 Qc2 Bc7 Nb6 Bd6 Rb3 Rc7 Na4 Qe7 Ndc5 Nxc5 Nxc5 g6 Nxb7 Rcxb7 Rb6 Rc7 Qb3 Qd7 Bxa6 Kg7 g3 Be7 Qc3 Bd8 Kg2 Ra8 Rcb1 Qe6 Qc2 Qe7 Bd3 Rxa5 Rxc6 Rxc6 Qxc6 Qf6 Qc2 h5 h4 Qe6 Rb7 Ra1 Be2 Re1 Ra7 Qb6 Rd7 Bf6 Bd3 Qe6 Ra7 Qg4 Qc7 Qe6 Qc3 Rd1 Qc2 Bxd4 Ra6 Qg4 exd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 cxd5 Nxd5 e3 Be7 Bb5 c6 Bd3 O-O Qc2 h6 O-O c5 Nxd5 Qxd5 e4 Qh5 dxc5 Bxc5 b4 Bd6 Bb2 Nc6 Qe2 Rad8 e5 Bb8 Rfd1 Ne7 Be4 Nd5 h3 Rd7 Rd4 Nf4 Qe3 Rxd4 Bxd4 Nd5 Qe1 f5 exf6 gxf6 Nh2 f5 Bf3 Qg6 Nf1 Rf7 Ng3 Nf4 Bxb7 Rxb7 Rd1 Nd5 Be5 Kh7 Nf1 f4 Bxb8 Rxb8 Qe2 Rg8 Qf3 Rg7 Re1 Qf5 Nd2 Nf6 Nc4 Nd5 Nd6 Qg6 Nb5 Ne7 Nd4 Nf5 Nxe6 Nh4 Qe4 Qxe4 Rxe4 Rxg2 Kf1 Rh2 Nxf4 Rh1 Ke2 Ng6 Nxg6 Kxg6 Re7 Rxh3 Rxa7 h5 Ra6 Kg7 a4 h4 Kf1 Ra3 Kg2 Kf7 b5 Ke7 f4 Kd7 f5 Kc7 Ra8 Rg3 Kh2 Rf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c5 d5 d6 Bg2 Bg7 Nh3 b5 cxb5 Qa5 Bd2 Qxb5 Bc3 Ba6 Na3 Qb6 O-O O-O Qd2 Nbd7 Rab1 Rab8 b3 Bb5 Rfe1 Ng4 Bxg7 Kxg7 Nc2 a5 f3 Ngf6 Nf2 Ba6 f4 Rfc8 e3 a4 g4 axb3 g5 Ne8 axb3 h5 gxh6 Kh7 Kh1 Nef6 e4 c4 e5 cxb3 Nd4 Rc4 Nc6 Rc2 Qe3 Qxe3 Rxe3 Rb6 exf6 Nxf6 Ne4 b2 Nxf6 exf6 Be4 Rc1 Re1 f5 Rbxc1 bxc1=Q Rxc1 fxe4 Re1 Bc4 Ne7 f5 Rg1 e3 Re1 Rb3 Kg2 Kxh6 Kf3 e2 Kf2 Rh3 Rg1 Rxh2 Kg3 Rh5 Nc8 Kg7 Nxd6 Ba6 Ne8 Kf7 Nc7 Bc4 Rc1 g5 Kf2 gxf4 Ne6 Rh2 Kg1 Rh4 Nd4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 Qa4 Bd7 Qh4 Nf6 e4 Bg4 Ne5 Be6 f4 Bg7 Qf2 O-O d4 Nc6 Nxc6 bxc6 h3 Qb8 Bd3 Rd8 Be3 Qb4 O-O Rab8 Rab1 Ne8 e5 f5 Rfc1 Bf8 b3 Ng7 Na4 Bd5 Nc5 Qb6 b4 e6 Na6 Rbc8 a4 Qb7 Qf1 Qa8 Bc4 Be4 Rb2 Kh8 Bd3 Bd5 Kh2 Be7 Rbc2 Rd7 b5 cxb5 Nxc7 Qb7 Nxd5 Rxc2 Rxc2 Rxd5 Bxb5 Rd8 Qc1 Qe4 Rc3 h6 Bc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Bxd2 Qxd2 O-O g3 d5 Bg2 Qe7 O-O Rd8 Rc1 c6 Qe3 Bd7 Nbd2 Be8 Nb3 Nbd7 Na5 Rab8 Rab1 Rdc8 cxd5 Nxd5 Qd2 c5 e4 N5b6 e5 h6 a3 Nd5 Nc4 Rd8 Qe1 cxd4 Nxd4 Nc5 Rd1 Nb6 Nd6 Nc8 Rbc1 Na4 Nxb7 Rxb7 Bxb7 Qxb7 Nxe6 Rxd1 Qxd1 Nab6 Rc7 Qe4 Qd8 Qe1 Kg2 Qe4 Kg1 Qe1 Kg2 Qe4 f3 Qe2 Kh3 Qb5 a4 Qxa4 Nc5 Qb5 Nd7 Qf1 Kg4 Nd6 Nf6 gxf6 exd6 h5 Kh4 Qe2 h3 Kg7 Re7 Qb5 f4 Bd7 Re5 Qf1 Rg5 fxg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O Bg5 h6 Bh4 c5 dxc5 Na6 a3 Bxc3 Qxc3 Nxc5 Bxf6 Qxf6 Qxf6 gxf6 b4 Na4 e3 b6 Nf3 Ba6 Nd2 Bb7 Bd3 Rac8 Rc1 d5 Ke2 dxc4 Nxc4 Rfd8 Rhd1 Ba6 Rd2 b5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Qe2 Nc6 O-O-O Nxd4 Rxd4 Be7 g3 Bd7 Bg2 h6 Bh4 Bc6 f5 O-O Rhd1 b5 g4 e5 R4d3 b4 Bxf6 bxc3 Bxe7 cxb2 Kb1 Qxe7 Rxd6 Rfc8 R1d2 Bb5 Qe3 Rab8 Rb6 Bc4 Rd1 Rxb6 Qxb6 Qa3 Qxb2 Bxa2 Ka1 Qa4 Qxa2 Qxa2 Kxa2 Rxc2 Kb3 Rxg2 Kc4 Rxh2 Kd5 f6 g5 hxg5 Ke6 g4 Rd8 Kh7 Kf7 Rh5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Nbd7 g4 h6 Be2 Qb6 Nb3 Qc7 Qg2 Rb8 Be3 b5 g5 hxg5 fxg5 Nh5 g6 Ne5 gxf7 Qxf7 Rf1 Qg6 Qxg6 Nxg6 Nd4 Ne5 a3 Bd7 O-O-O Rc8 Bg5 Nf6 Bf4 Nc4 Bxc4 Rxc4 e5 dxe5 Bxe5 Ng4 Bf4 Bc5 Ne4 e5 Nxc5 Rxc5 Rfe1 O-O Nb3 Rxc2 Kxc2 Bf5 Kc3 exf4 Rd2 Rc8 Kb4 Ne3 Rc1 Rxc1 Nxc1 f3 Kc5 Ng4 Ne2 fxe2 Rxe2 Kf7 Kb6 b4 axb4 Bd3 Rg2 Ne5 Kc5 Bb5 Kd4 Nd3 Kc3 Kf6 Rg3 Nf4 Rg4 Ne6 Kd2 g5 Ke3 Kf5 Rg3 Nf4 Rf3 Ke5 Rg3 Nd5 Kf2 Kf5 Rf3 Kg4 Rg3 Kh4 Kf3 Nf6 Kg2 Nh5 Re3 Nf4 Kg1 Kg4 Rg3 Kf5 Rf3 g4 Re3 Nd5 Rb3 Kf4 Kf2 Nf6 Ra3 Ne4 Kg2 Be2 Rb3 Bf1 Kg1 Bc4 Ra3 Ng5 Rc3 Bd5 Ra3 Be4 Kf2 Bb7 Rd3 Nh3 Ke1 Bc6 Rd4 Kf3 Rd3 Ke4 Rg3 Kf4 Ra3 Bb5 Rc3 Ng5 Rc8 Ke3 Rh8 Nf3 Kd1 Be2 Kc2 Nxh2 Rxh2 g3 Rh3 Kf2 Kd2 g2 Rh2 Bb5 Rh6 g1=Q Rf6 Kg2 Rg6 Kf1 Rxg1 Kxg1 Ke1 Kg2 Kd1 Kf3 Kd2 Ke4 Kc3 Ke3 Kc2 Ke2 Kc1 Bd3 b3 Ke1 Kb2 Kd2 Ka1 Kc2 Ka2 Kc1 Ka1 Bb1
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 c5 g3 g6 Bg2 Bg7 d4 cxd4 Nxd4 Nc6 Nb3 d6 Nc3 Nf6 O-O O-O e4 Bg4 f3 Be6 Kh1 b5 a4 bxa4 Rxa4 Nd7 f4 Nb6 Ra1 a5 Nd5 a4 Nd2 Bd7 Ra3 Na5 f5 Nxd5 exd5 Qb6 fxg6 fxg6 b3 Rxf1 Nxf1 axb3 cxb3 Rb8 Bg5 Nxb3 Bxe7 Nd4 Rd3 h6 Ne3 Re8 Nc4 Qb4 Nxd6 Rxe7 Ne4 Nf5 Qf3 Qe1
Result: 0-1
Processing game...
FEN: 
Moves: c4 e6 Nc3 Bb4 Nf3 Nf6 g3 d5 a3 Be7 Bg2 O-O d4 dxc4 Ne5 c5 dxc5 Qc7 Nxc4 Qxc5 Qb3 Nc6 Be3 Nd4 Bxd4 Qxd4 O-O Rb8 Rfd1 Qc5 Rac1 Rd8 Ne4 Nxe4 Bxe4 Rxd1 Qxd1 Qc7 b4 Qd7 Qc2 g6 Na5 Bd8 Rd1 Qe7 Qd3 Bc7 h4 e5 h5 gxh5 Bxh7 Kg7 Bf5 Qf6 Bxc8 Bxa5 Rc1 Bb6 e3 Qg6 Qd5 Qg5 Kg2 h4 Rc4 Rxc8 Rxc8 h3 Kh1 Qf5 Rc4 f6 Qxb7
Result: 1-0
Processing game...
FEN: 
Moves: d4 g6 c4 Bg7 Nc3 d6 Nf3 Nf6 e4 O-O Be2 e5 d5 a5 h4 Na6 Nd2 Nc5 g4 a4 h5 gxh5 g5 Ng4 Nf1 f5 f3 Nf2 Kxf2 fxe4 Kg2 a3 Rxh5 exf3 Bxf3 e4 Be2 Bf5 Qd2 Bg6 Rh3 Qd7 Rg3 Qf7 Qe1 Rae8 Be3 Nd3 Qd2 axb2 Rb1 Be5 Rg4 Nf4 Bxf4 e3 Bxe3 Bxc3 Qxc3 Bxb1 Rf4 Rxe3
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 d4 Nf6 e4 d6 Bd3 O-O Nge2 c5 d5 e6 O-O exd5 cxd5 Na6 h3 Re8 Bg5 h6 Bh4 g5 Bg3 c4 Bc2 Nc5 f3 Bd7 Bf2 Nh5 Bd4 Be5 b4 cxb3 axb3 a5 Qd2 Qf6 Bxe5 dxe5 Na4 Qd6 Nxc5 Qxc5 Kh2 Nf4 Bd3 Rec8 Nxf4 gxf4 g3 Qe3 Rad1 Qxd2 Rxd2 fxg3 Kxg3 Rc3 Rb1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3 Qxc3 Ne4 Qc2 c5 dxc5 Nc6 cxd5 exd5 Nf3 Bf5 b4 O-O Bb2 b6 b5 bxc5 bxc6 Qa5 Nd2 Rab8 c7 Rb3 Rd1 c4 e3 Ng3 Qxb3 cxb3 hxg3 Bc2 Rc1 d4 Bd3 Rc8 Bxc2 bxc2 Rxc2 dxe3 fxe3 Rxc7 Rh5 Qxh5 Rxc7 h6 Bd4 a6 Nf3 Qd5 Rc3 Qa2 Kf1 Qb1 Kf2 Kh7 Ne5 f6 Nd3 Qa2 Kg1 Kg6 Nf4 Kf5 Kh2 Kg4 e4 Qd2 Rd3 Qc2 Re3 a5 e5 fxe5 Rxe5 a4 Nd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 Bf4 Be6 e3 h6 h4 Nb4 Qd2 N4d5 e4 Nxf4 Qxf4 Qd6 Qd2 c6 e5 Qc7 h5 g5 Ne4 Rad8 Qc2 Qc8 Qc5 f5 exf6 exf6 Qa5 Bf7 Qxa7 Nd5 Rfe1 Nb4 Qa4 Nd3 Re2 g4 Nh4 f5 Nc5 Nxc5 dxc5 Bxh5 Re7 Bf6 Qb3 Bf7 Qxb7 Bxe7 Qxe7 Qd7 Qxd7 Rxd7 Nxf5 Rd2 Nxh6 Kg7 Nxg4 Be6 Ne3 Rfxf2 Bxc6 Rfe2 Ng2 Bh3 Nf4 Re3 Nxh3 Rxg3 Kf1 Rxh3 b4 Rb2 b5 Rc3 a4 Rxc5 a5 Rf5 Kg1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nf6 Nc3 g6 e4 Bg7 d4 cxd4 Nxd4 d6 Be2 Nc6 Be3 O-O O-O Bd7 Qd2 Nxd4 Bxd4 Bc6 Bd3 Nd7 Bxg7 Kxg7 Rfe1 a5 Re3 Nf6 Rd1 a4 Nd5 e6 Nb4 Qa5 Bc2 Rfd8 Qc3 e5 h3 Qc5 b3 axb3 axb3 Ra3 Qd2 Qa5 Nxc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 g3 d5 Bg2 Nf6 d4 Be7 Nf3 c6 Qc2 O-O O-O Nbd7 b3 b6 Rd1 Bb7 Nc3 Rc8 e4 c5 exd5 exd5 Bb2 dxc4 d5 Ne8 Nb5 a6 Na7 Rc7 Nc6 Bxc6 dxc6 Rxc6 Ne5 Rd6 Nc6 Rxc6 Bxc6 Nd6 bxc4 Nf6 Be5 Qc7 Bg2 Rd8 Rd3 Ng4 Bf4 Bf6 Rad1 Bd4 Rxd4 cxd4 Rxd4 Qe7 h3 Ne5 c5 bxc5 Qxc5 f6 Qa3 h5 Qxa6 Ndf7 Rxd8 Nxd8 a4 Ng6 Be3 h4 gxh4 Qc7 Bd5 Kh8 Qc4 Qd7 Qb5 Qe7 h5 Nh4 a5 Nf5 Qc5 Qe8 a6 Nxe3 Qxe3 Qa4 Qe4 Qa1 Kg2 Qxa6 Qe8 Kh7 Bg8 Kh8 Bf7
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 Nf3 c5 c3 Nc6 e4 d5 e5 Bd7 Be2 Nh6 O-O cxd4 cxd4 Nf5 Nc3 Be7 a3 Rc8 Bf4 g5 Be3 g4 Ne1 h5 Nc2 f6 exf6 Bxf6 Bd3 Nce7 Qd2 O-O Bg5 Ng6 Bxf6 Qxf6 Rae1 Nfh4 Bxg6 Qxg6 Kh1 Rf5 Ne3 Rf7 Nc2 Rcf8 Nb4 Qg7 Re5 Rf5 Nd3 Ng6 Rxf5 Rxf5 Qe3 Qf6 Re1 b6 Kg1 h4 Ne2 Bb5 Nec1 Kf7 Ne5 Nxe5 dxe5 Qg6 Nb3 Rf3 Qc1 Rxb3 Qc7 Kg8 Qd8 Be8 Qxh4 Rxb2 h3 gxh3 Qxh3 Rb1 Rxb1 Qxb1 Kh2 Qf5 Qg3 Kf8 Qe3 b5 f3 a5 g4 Qh7 Kg3 b4 axb4 axb4 Qf4 Qf7 Qxb4 Kg7 Qd2 Qe7 Qa5 Bg6 Qb6 Qd7 Qd4 Qc7 Qf4 Qd8 Qd4 Qa5 Qf4 Qd8 Qd4 Bf7 Kg2 Qg5 Kg3 Qc1 Kf2 Qc2 Kg3 Qc4 Qe3 Kg6 Kh4 Qb4 Qg5 Kh7 Qf6 Qf8 Kg3 Qg7 Qh4 Qh6 Qe7 Kg7 Qd8 Qg6 Kf4 Be8 Kg3 Qf7 Qg5 Kf8 Qh6 Ke7 Qe3 Qg7 Qf4 Kd8 Qe3 Qc7 Qg5 Kd7 Qg7 Kc8 Qf6 Qd7 Qf8 d4 Qc5 Kd8 Qb6 Ke7 Qb4 Kf7 Qd2 Qd5 Qg5 d3 Qf6 Kg8 Qg5 Kf8 Qf6 Bf7 Qh6 Ke8 Qh8 Kd7 Qg7 d2 Qxf7 Kc6 Qe8 Kc5 Qe7 Kc4 Qc7 Kd3 Qh7 Kd4 Qa7 Kc3 Qc7 Qc4 Qa5 Kc2 Kh4 d1=Q Kg5 Qd2 Qxd2 Kxd2 Kf6 Ke3 g5 Qf4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c6 Bf4 d6 h3 Qb6 b3 c5 dxc5 Qxc5 c4 g6 Nc3 Bg7 Rc1 Bf5 Be3 Qa5 Bd2 Qd8 Nd4 Be4 e3 Nc6 Nde2 Bd3 Nf4 Bxf1 Kxf1 O-O g3 e6 Kg2 d5 cxd5 exd5 Qf3 d4 Ncd5 Nxd5 Qxd5 Qe7 Rhe1 Rad8 Qc5 Qd7 e4 Ne5 Red1 b6 Qc7 Qb5 Qxa7 g5 Qa4 Qxa4 bxa4 gxf4 gxf4 Nd3 Rc6 Nb2 Rdc1 Nxa4 f5 Rfe8 f3 Ra8 Kh1 Nc5 Rg1 Kh8 Rc7 Rf8 Bf4 Rac8 Re7 Rce8 Rc7 Nd3 Bd6 Rg8 Rxf7 Be5 Rxg8 Rxg8 Rd7 Nf2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 Bg5 h6 Bh4 Qa5 O-O Qh5 Qxh5 Nxh5 f3 b6 Rfd1 Bb7 Bc4 Be7 Bf2 Rc8 Bf1 O-O Nb3 Rc6 a4 Nhf6 Rd2 Rfc8 Nd4 R6c7 a5 b5 Bxb5 axb5 Ndxb5 Rc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 f3 Ng8 Nb3 axb4 axb4 Ndf6 Bd2 Nh5 g3 Nhf6 Rf2 Nh6 Ra1 Rxa1 Qxa1 Nf7 Qc1 f4 g4 h5 h3 Nh7 Be1 Bf6 Kg2 Kg7 Rf1 Nhg5 Rh1 Nxh3 Rxh3 Ng5 Rh2 hxg4 fxg4 Rh8 Bh4 f3 Bxf3 Nxf3 Bxf6 Qxf6 Rxh8 Kxh8 Kg3 g5 Ne2 Kg7 Qh1 Nd4 Nbxd4 exd4 Qd1 Qe5 Kf3 Qf6 Kg3 Qe5 Kf3 Qf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 d6 Nf3 Nbd7 g3 b6 Bg2 Bb7 O-O Qe7 b4 c5 Bb2 cxd4 Qxd4 Rfc8 Rfd1 Ne8 e4 Rc7 Nd2 Rac8 Rac1 Qf6 Qxf6 Nexf6 f3 Kf8 Bf1 Ke7 Be2 g5 Kf2 g4 fxg4 Nxe4 Nxe4 Bxe4 g5 Ne5 Rd4 Bg6 h4 Nc6 Rd2 Ne5 h5 Be4 Rcd1 Nxc4 Bf6 Ke8 Bxc4 Rxc4 Rxd6 Bd5 Re1 Kf8 Re2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Bb4 Nbd2 Nc6 O-O Nf6 e5 Nd5 Nb3 Nb6 Bg5 Be7 Bxe7 Qxe7 Bb5 Bd7 Bxc6 Bxc6 Nfxd4 Bd5 Qg4 g6 Rfe1 O-O Nf5 Qd7 Nh6 Kg7 Qg5 f6 exf6 Rxf6 Re7 Qxe7 Nf5 Rxf5 Qxe7 Rf7 Qe5 Kg8 Nc5 c6 b3 Nd7 Nxd7 Rxd7 Rd1 a5 h4 Rf7 Rd4 Rf5 Qe1 b5 Rd3 a4 Re3 axb3 axb3 Raf8 f3 h5 Kh2 R5f6 Kg3 Rf5 Qd2 Kg7 b4 R5f7 Qd4 Kg8 Qb6 Kg7 Kf2 Kh7 Kg3 Kg7 Qd4 Kg8 Ra3 Kh7 Ra6 Kg8 Rb6 Kh7 Kh2 Rf4 Qe3 R4f7 Qc3 Rf4 Kg3 R4f7 Ra6 Rf5 Ra1 R5f7 Re1 Rf5 Re3 R5f7 Qc5 Kg7 Kf2 Rf5 Qd4 Kh7 Re7 R5f7 Re5 Rf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 d5 cxd5 Nxd5 e4 Nb4 Bb5 N8c6 d4 cxd4 a3 dxc3 Qxd8 Kxd8 axb4 cxb2 Bxb2 f6 e5 Bg4 Bxc6 bxc6 Nd4 fxe5 Nxc6 Kc7 Nxe5 Bh5 O-O Be8 Rfc1 Kb7 Nc4 e5 Bxe5 h5 Na5 Kb6 Bc7 Ka6 Nc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 h6 Be3 e6 Qd2 exd5 cxd5 Re8 f3 h5 a4 a6 a5 Nh7 Bd1 Nd7 Nge2 Ne5 b3 Qh4 Bf2 Qf6 Qe3 h4 h3 g5 O-O Bd7 Kh1 Nf8 Bg1 Nfg6 Bh2 c4 Rb1 Rac8 Qa7 Nf4 bxc4 Nxc4 Rxb7 Nd2 Rg1 Nxh3 e5 Rxe5 Rxd7 Nxg1 Qxg1 Ree8 Rxd6 Qf5 Ba4 Qd3 Bxe8 Rxe8 Rc6 h3 Bg3 hxg2 Qxg2 Nxf3 d6 Re6 Qf2 g4 Rc8 Kh7 Nf4 Rh6 Kg2 Ne1 Kg1 Nf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 h6 O-O Ng4 Bc1 Nc6 d5 Ne7 Ne1 f5 Bxg4 fxg4 Nc2 g5 Ne3 Rf4 Bd2 Bd7 b4 Qf8 Qc2 Qf7 Rac1 Rf8 Nb5 Rxf2 Nxc7 Qf4 Nf5 Bxf5 Rxf2 Qxf2 Kxf2 Bxe4 Ke2 Bxc2 Rxc2 Rc8 c5 e4 Nb5 dxc5 d6 Nc6 Rxc5 Rd8 Rd5 a6 Nc7 Be5 b5 axb5 Nxb5 Kf7 Ke3 Bxh2 Kxe4 Ke6 Bc3 Bxd6 Rxd6 Rxd6 Nxd6 Kxd6 Kf5 g3 Bg7 Ne7 Kg4 Ng8 Kf5 Ne7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6 Rd1 a6 Ng5 Bxg2 Kxg2 Nc6 Qf4 Ra7 Nge4 Rd7 Be3 Ne5 Nxf6 Bxf6 Na4 Qa8 Kg1 Qc6 Nxb6 Rb7 c5 dxc5 Nc4 Nxc4 Qxc4 Bxb2 Rab1 O-O Qxc5 Qa4 Rd2 Rb5 Qc6 Bd4 Rxb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 b6 g3 Bb7 Bg2 Be7 d4 Ne4 Bd2 Bf6 O-O O-O Rc1 Nxd2 Qxd2 d6 d5 e5 b4 a5 a3 Nd7 e4 g6 h4 Ba6 Nb5 h5 Rc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 e3 O-O Bd3 c5 Nf3 Nc6 O-O cxd4 Nxd4 Bg4 Qa4 Nxd4 Qxd4 Qd7 h3 Be6 Rfd1 Rfc8 Be5 h6 Rd2 Qd8 Bc2 Qa5 Qd3 Kf8 Rad1 Rd8 Qd4 Rac8 Bb3 Kg8 Qf4 Rd7 Qg3 Nh5 Qf3 Nf6 Rd3 a6 Bxf6 Bxf6 Bxd5 Rcd8 e4 Bxc3 Bxe6 Rxd3 Qxf7 Kh8 Rxd3 Bf6 Rxd8 Qxd8 Qxb7 Qd1 Kh2 Qd6 g3 Qxe6 b3 Qd6 Kg2 Bd4 h4 Qf6 f4 Qd6 e5 Qc5 Qxa6 Qd5 Kh2 Qf3 Qc8 Kh7 Qc2 Kh8 Qg2 Qd3 h5 Bc5 e6 Kg8 a4 Qxb3 Qa8 Bf8 Qe8 Qc2 Kh3 Qb1 Qf7 Kh7 Qg6 Qxg6 hxg6 Kxg6 a5 h5 f5 Kf6 a6 Bc5 Kh4 g6 e7 Kxe7 fxg6 Kf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 Nc6 d4 cxd4 Nxd4 e6 g3 Qb6 Ndb5 d5 Bg2 d4 Na4 Qa5 Bd2 Bb4 Nc5 O-O Nd3 Bxd2 Qxd2 Qxd2 Kxd2 Rd8 c5 Ne8 Na3 f6 f4 Bd7 Nc4 Rab8 b4 Ne7 a4 Bc6 Bf3 a6 Rhb1 Nc7 Nd6 Nf5 b5 axb5 axb5 Nxd6 bxc6 Nc4 Kc1 bxc6 Rxb8 Rxb8 Ra4 Ne3 Bxc6 e5 fxe5 fxe5 Nxe5 Ne6 Ra5 Rc8 Kd2 Kf8 Kd3 Ke7 Ra7 Kf6 Nd7 Kg5 Bf3 Nf5 h4 Kg6 Ra6 Kf7 Bd5 Re8 Ne5 Kf6 Nf3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d5 cxd5 Nxd5 O-O Nb6 Nc3 Nc6 d5 Na5 Bf4 c6 dxc6 Qxd1 Raxd1 Nxc6 Nb5 Bg4 b3 e5 Be3 e4 Nfd4 Nxd4 Nxd4 Nd5 h3 Nxe3 fxe3 Bd7 g4 Rae8 Rf4 f5 Nxf5 Bxf5 gxf5 gxf5 Rff1 Bh6 Kf2 Bg5 Rd5 Rd8 Re5 Rde8 Rd5 Kg7 Kg3 Kg6 h4 Bxe3 h5 Kxh5 Rfxf5 Rxf5 Rxf5 Kg6 Rd5 b6 Bh3 h6 Bg4 Re7 Rd6 Kg5 Re6 Bf4 Kh3 Rc7 Rxe4 Rc2 Bf3 Rxa2 Ra4 Rb2 Rxa7 Rxb3 Kg2 Rb2 Kf2 Be5 Ke3 Rb4 Rd7 Kf5 Kd3 b5 Rf7 Bf6 Rb7 Rb3 Kc2 Rb2 Kd3 Rb3 Kc2 Rc3 Kd1 Rc5 e3 Bc3 Be2 Rd5 Kc2 b4 Rb5 Rxb5 Bxb5 h5 Kd3 Kg4 Ke2 Kg3 Kf1 Kh2 Bc4 h4 Kf2 h3 Be6 Bf6 Bd5 Bh4 Kf1 Bg5 Kf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 e6 cxd5 exd5 Nc3 c6 Qc2 Na6 a3 Nc7 Bg5 g6 e3 Bf5 Bd3 Bxd3 Qxd3 Be7 O-O O-O b4 Ne4 Bf4 Nxc3 Qxc3 Bd6 Bxd6 Nb5 Qb3 Nxd6 a4 a6 Ne5 Re8 Rfe1 Qg5 h3 Kg7 Qc2 Re6 Rac1 Rae8 Qb1 Qh5 Qb3 f6 Nd3 g5 Qd1 Qg6 Qc2 R6e7 Red1 h5 Qb1 h4 Qc2 g4 Nf4 Qxc2 Rxc2 g3 Rd3 Kh6 Kf1 Kg5 Ne2 Nc4 Rcc3 Nb2 Rd2 Nxa4 Rb3 Nb6 Ng1 Nc4 Nf3 Kh5 Rdd3 a5 bxa5 Ra8 Rd1 Rxa5 Re1 b5 Re2 Ra1 Re1 Rea7 fxg3 Rxe1 Kxe1 Ra1 Ke2 hxg3 Ne1 Ra2 Kd1 Rd2 Kc1 Re2 Kd1 Rxe3 Rxe3 Nxe3 Ke2 Nf5 Nc2 Nh4 Nb4 Nxg2 Kf3 Nh4 Kxg3 Nf5 Kf4 Nxd4 Ke3 Nf5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Bd3 Nbd7 Nf3 g6 O-O Bg7 a4 O-O Qe2 Qc7 Kh1 b6 e5 dxe5 fxe5 Ng4 Bf4 Bb7 Be4 Bxe4 Qxe4 Ngxe5 Rae1 f6 Bxe5 Nxe5 Nxe5 fxe5 Rxf8 Rxf8 Qe2 Qc6 h3 e4 Qxa6 Bxc3 bxc3 Qxc3 Rxe4 Qxc2 Rxe6 Qc1 Kh2 Rf1 Qa8 Kg7 Qe4 Qc7 Qe5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 Nc6 Qd2 O-O Bc4 Bd7 h4 Ne5 Bb3 Rc8 O-O-O Nc4 Bxc4 Rxc4 h5 Nxh5 g4 Nf6 Bh6 Nxe4 Qe3 Rxc3 bxc3 Nf6 Bxg7 Kxg7 Rh2 Rh8 Nb3 Bc6 g5 Nh5 f4 Re8 f5 Qb6 Nd4 Qc5 Re1 Bd7 Qf3 Bc6 Qe3 Bd7 Qf3 Bc6 Qf2 Kg8 Re3 Bd5 Rxh5 gxh5 Qh4 Qc4 Qxh5 Qf1 Kb2 e5 Qh6 Kh8 g6 fxg6 fxg6 Re7 Rf3 Qc4 Qf8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 d6 Nf3 Nf6 e5 dxe5 Nxe5 e6 Bb5 Nbd7 d4 cxd4 Qxd4 a6 Bxd7 Nxd7 Be3 Qf6 f4 Bc5 Qd2 Bxe3 Qxe3 Nxe5 fxe5 Qh4 g3 Qh5 Qc5 f6 Ne4 Qxe5 Qxe5 fxe5 O-O-O Ke7 Rhf1 b5 Ng5 h6 Nf7 Rf8 Nxe5 Rxf1 Rxf1 Kd6 Nf7 Ke7 Ne5 Kd6 Nf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 g4 h6 h4 Nc6 Rg1 h5 gxh5 Nxh5 Bg5 Nf6 Be2 Bd7 h5 a6 Qd2 b5 a3 Be7 Be3 Nxh5 O-O-O Nf6 Rxg7 Qb8 Nxc6 Bxc6 Qd4 e5 Qd3 Kf8 Rdg1 Qc8 f4 Qh3 R7g3 Qh2 Bf3 Rg8 Rxg8 Nxg8 Ne2 Qh3 Ng3 exf4 Bxf4 Qh4 Qe3 Rc8 Nh5 Qh3 Bh6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O O-O d5 cxd5 Nxd5 d4 Nb6 Nc3 Nc6 e3 Re8 d5 Na5 Nd4 Bd7 Qe2 c6 dxc6 Nxc6 Nxc6 Bxc6 Bxc6 bxc6 Rd1 Qc8 Bd2 Qe6 Be1 h5 Rac1 h4 b3 Rad8 gxh4 Nd7 Kg2 Nc5 Rxd8 Rxd8 Rd1 Rxd1 Qxd1 Qf5 f3 Nd3 Ne4 Qd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Nbd7 O-O O-O Re1 Bb7 e4 dxe4 Nxe4 c5 Ned2 a6 Qe2 cxd4 Nxd4 Bxg2 Kxg2 Qc7 N2f3 Qb7 b4 Rfc8 Rac1 Bf8 Kg1 Rc7 a3 Re8 Bb2 e5 Nf5 e4 N3h4 Ne5 Red1 h6 Bxe5 Rxe5 Ne3 Re6 Nhg2 Nd7 Nf4 Re8 Nfd5 Rc6 Qg4 Rce6 h4 Qb8 Kg2 Qe5 Qf5 g6 Qxe5 Rxe5 Nc7 Rd8 Rc2 a5 Rcd2 axb4 axb4 Re7 Ncd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 e6 Nc3 dxc4 e4 Bb4 Bg5 c5 Bxc4 cxd4 Nxd4 Bxc3 bxc3 Qa5 Bb5 Bd7 Bxf6 gxf6 Qb3 a6 Be2 Nc6 O-O Qc7 Rab1 Na5 Qa3 Rc8 Rfd1 Qxc3 Qd6 Qc7 Nf5 exf5 Qxf6 O-O Rd3 f4 Rd5 h6 Qxh6 f5 Rb6 Bc6 Rxa5 Qh7 Qxf4
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 c5 g3 Bg7 Bg2 Nc6 e3 d6 Nge2 Nf6 O-O O-O d3 Bd7 h3 a6 b3 Rb8 Bb2 b5 Qd2 Nb4 Rfd1 Re8 e4 Nc6 Nd5 e5 Kh2 h5 f4 bxc4 dxc4 Nxd5 Qxd5 Nd4 Nxd4 cxd4 Qxd6 h4 g4 Bf8 Qxa6 Re6 Qa7 Ra8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O e4 d6 Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5 f3 f4 Bf2 g5 b4 Nf6 c5 Ng6 cxd6 cxd6 Rc1 Rf7 a4 Bf8 a5 Bd7 Nb5 g4 Nc7 g3 Nxa8 Nh5 Kh1 gxf2 Rxf2 Ng3 Kg1 Qxa8 Bc4 a6 Qd3 Qa7 b5 axb5 Bxb5 Nh1
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Bxd2 Qxd2 O-O g3 d5 Bg2 Qe7 O-O Rd8 Qc2 Na6 a3 dxc4 Qxc4 c5 dxc5 Qxc5 Nbd2 Bd7 b4 Qxc4 Nxc4 Bb5 Rfc1 Rac8 Nfe5 b6 Bb7 Rc7 a4 Nxb4 axb5 Rxb7 Nd6 Re7 Nc8 Ree8 Nxa7 Nbd5 Nac6 Ra8 Nc4 Ne4 Kf1 Rxa1 Rxa1 Nc7 Na7 Rd8 Nxb6 Nc3 Ra5 Rd1 Kg2 Rb1 Nd7 f6 b6 N7d5 Nc8 Nxe2 b7 Rg1 Kh3 Nef4 Kg4 h5 Kf3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 O-O Nc6 c4 Ne5 d3 e6 Nc3 Ne7 Ne1 g6 Nc2 Bg7 Be3 O-O f4 N5c6 d4 cxd4 Nxd4 f5 exf5 Nxf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 c5 dxc5 O-O e3 Nbd7 Bd3 Qa5 Ne2 Bxc3 Qxc3 Qxc3 Nxc3 Nxc5 Bc2 g5 Bg3 Be6 f3 a6 h4 Rfc8 hxg5 hxg5 O-O-O Ncd7 Kb1 Kg7 Bb3 Rc5 e4 dxe4 Bxe6 fxe6 Nxe4 Ra5 Rhe1 Nf8 Nd6 b6 Bf2 Nd5 Bd4 Kg6 Nc4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Qe2 Nc6 O-O-O Nxd4 Rxd4 Be7 e5 dxe5 fxe5 Nd5 Bxe7 Nxe7 Ne4 O-O Qh5 Ng6 Ng5 h6 Nf3 b5 Bd3 Bb7 Bxg6 fxg6 Qxg6 Bd5 Kb1 Rac8 Rc1 Qe7 Rg4 Rc4 Rxc4 bxc4 c3 Qa7 Qc2 Bxf3 gxf3 Qe3 Qa4 Qd3 Ka1 Rxf3 a3 Qd5 Qxa6 Rh3 Rg1 Rxh2 Qc8 Kh7 Qc7 Rg2 Rxg2 Qd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 c5 Bg2 cxd4 Nxd4 O-O Nc3 Nc6 O-O Nxd4 Qxd4 d6 Qd3 a6 h3 Nd7 b3 Rb8 Be3 b5 cxb5 axb5 Rac1 b4 Na4 Qa5 Qd2 Bb7 Bxb7 Rxb7 Rfd1 Nf6 Rc4 h5 Qc2 Rfb8 f3 Qe5 Bf2 Qe6 g4 hxg4 hxg4 Rb5 Rc1 Bh6 Rc8 Kg7 Rxb8 Rxb8 Rd1 Nxg4 Bd4 Nf6 Kg2 Rb5
Result: 0-1
Processing game...
FEN: 
Moves: d4 e6 e4 d5 Nd2 b6 Ngf3 Bb7 Bb5 c6 Bd3 Be7 O-O dxe4 Nxe4 Nf6 Nxf6 Bxf6 Bf4 O-O Qe2 Qd5 Be4 Qh5 a4 Nd7 a5 bxa5 Bd6 Rfc8 Qd2 Bd8 Ne5 Nxe5 dxe5 Ba6 Rfe1 f5 exf6 Bxf6 Rxa5 Bb5 c4 Rd8 Qb4 Rac8 cxb5 c5 b6 Rxd6 bxa7 Rdd8 Qa4 Be5 a8=Q Qxh2 Kf1 Rxa8 Rxa8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Nd3 f5 Bd2 Kh8 Rc1 Nf6 f3 c5 g4 a6 Nf2 h6 h4 fxg4 fxg4 Neg8 Kg2 Nh7 Rh1 Bf6 g5 hxg5 h5 Qe8 b4 cxb4 Na4 Bd8 Bxb4 Bd7 hxg6 Qxg6 c5 g4 c6 g3 Nd3 bxc6 dxc6 Rc8 Bf3 Rxf3 Qxf3 Bg4 Qxg3 Qxe4 Kg1 Nf6 Bxd6 Qd4 Nf2 Qxd6 Nxg4 Qd4 Nf2 Qxa4 Qxe5 Rc7 Rh2 Rg7 Rg2 Bc7 Qf5 Qxa2 Qc8 Qg8 Qxg8 Kxg8 Rxg7 Kxg7 Nd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nc6 Nc3 e5 g3 g6 Bg2 Bg7 d3 d6 e4 Be6 Nge2 Qd7 Nd5 Nce7 d4 c6 Ne3 Bh3 O-O Bxg2 Kxg2 exd4 Nxd4 h5 a4 Nh6 Ra3 O-O-O a5 h4 a6 b6 Nf3 hxg3 fxg3 f6 c5 dxc5 Qb3 Kb8 Rd1 Qc8 Nc4 Rxd1 Qxd1 Nf7 Rd3 g5 Qb3 Qe6 Be3 Nc8 Nxb6 Qxb3 Nd7 Kc7 Rxb3 Kxd7 Rb7 Ke6 Bxc5 Bf8 Nd4 Ke5 Nxc6 Kxe4 Bxa7 Nfd6 Rc7 Nxa7 Rxa7 Nc4 Ra8 Bg7 Ra7 Ne3 Kg1 Rg8 Re7 Kd3 a7 Rc8 Kf2 Nd1 Kf3 g4 Kxg4 f5 Kf3 Bxb2 Rd7 Kc2 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 g6 e4 Bg7 d4 d6 c4 Nf6 Nc3 O-O Be2 e5 O-O Nc6 Be3 Ng4 Bg5 f6 Bc1 f5 Bg5 Bf6 Bxf6 Nxf6 dxe5 dxe5 Qxd8 Rxd8 Nd5 Nxe4 Nxc7 Rb8 Rfd1 Bd7 Bd3 Rbc8 Nd5 Nc5 Bf1 Be6 b4 Bxd5 cxd5 Nxb4 Nxe5 Ne4 Bc4 Nc3 Rd2 Nbxd5 g3 Kg7 Kg2 Ne3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 c5 dxc5 Nc6 e3 g5 Bg3 Ne4 Nf3 Qf6 Bb5 Nxc3 Bxc6 bxc6 a3 g4 Be5 Ne4 axb4 Qf5 Bxh8 gxf3 Rg1 Qg4 Qd1 Ng5 Qd4 Ne4 Qe5 Be6 Qf4 Qg6 Qxf3 f6 Qf4 Kf7 f3 Ng5 Kd2 Qf5 h4 Qxf4 exf4 Nh7 g4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O d4 d6 O-O Nbd7 Nc3 e5 h3 c6 e4 Qb6 c5 dxc5 dxe5 Ne8 Na4 Qa6 Bf4 Nc7 Qc2 Ne6 Rfd1 Re8 Rd6 Qa5 Rad1 Nb6 Nxb6 axb6 a3 Qa4 Qe2 b5 Qe3 b4 axb4 Qxb4 R6d2 Ra2 Rb1 c4 Rc2 b5 Bh6 Qc5 Qc1 Nd4 Bxg7 Kxg7 Nxd4 Qxd4 b3 Rxc2 Qxc2 c3 Rd1 Qc5 b4 Qxb4 Rd3 c5 Rxc3 c4 f4 Qc5 Kh2 Qd4 Rf3 b4 Qa4 c3 Rxc3 Bd7 Rc4 Bxa4 Rxd4 Rb8 Bf1 Bc2 Bc4 b3 Bxb3 Rxb3 g4 Re3 f5 gxf5 exf5 Rxe5 Rd2 Ba4 Kg3 Re3 Kh4 Bb5 Rd5 Bd3 Rc5 h6 Rc3 Rf3 Rb3 Be2 Rb2 Bf1 Rh2 Kf6 Rh1 Ke5 Rh2 f6 Rh1 Ke4 Rh2 Kf4 Rh1 Bg2 Rh2 Rg3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Be7 e4 dxc4 Bxc4 c5 dxc5 Qxd1 Kxd1 Bxc5 Nb5 Bb6 e5 Nd5 Nd6 Ke7 f4 Nc6 Nf3 Ne3 Bxe3 Bxe3 g3 Rd8 b4 f6 Ke2 Bd4 b5 Bxa1 bxc6 fxe5 Nxc8 Raxc8 Rxa1 exf4 Rb1 Rxc6 Rxb7 Kf6 Rb4 fxg3 hxg3 h5 Ke3 e5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5 Nc6 fxe6 fxe6 Nxc6 bxc6 Be2 Be7 O-O O-O Rb3 Qc5 Be3 Qe5 Bf4 Qc5 Kh1 Ng4 h3 e5 Na4 Qa7 Bc4 Kh8 hxg4 exf4 Nb6 d5 exd5 cxd5 Bxd5 Rb8 Nxc8 Rbxc8 Rh3 Qb6 Re1 Bg5 Re6 Qd8 c4 Rb8 Qd3 Bh4 Be4 Qg5 Bxh7 Rfd8 Qc2 f3 Rxf3 Rd2 Qe4 Rd1 Kh2 Re1 Qf5 Rxe6 Qxe6 Kxh7 Qe4 g6 Rh3 Kg7 Qd4 Kg8 Qe4 Qf6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 c6 Bd3 e5 d5 b5 cxb5 cxd5 exd5 e4 Nxe4 Nxd5 Bg5 Qb6 Qd2 Nd7 Bc4 N5f6 Nxf6 Bxf6 Bxf6 Nxf6 Ne2 Re8 O-O-O d5 Bd3 a6 bxa6 d4 Kb1 Re3 Bc4 Bxa6 Bxa6 Rxa6 Nxd4 Re8 Ne2 Rb8 Nc3 Qb4 Rhe1 Rd6 Qc2 Rdb6 Re2 Qf4 h3 Rc6 Qd2 Qf5 Ka1 Rb7 Qh6 Rc8 Red2 Qa5 Qe3 Kg7 g4 Re8 Qd4 Rd7 Qf2 Rc7 Rd3 Ra8 Qd2 h6 Rd6 Rc4 Rd4 Rac8 Kb1 Qe5 f4 Qe6 Qe2 Rxd4 Rxd4 Qb6 Qd2 Qa6 Qd3 Qc6 a3 Qg2 Rd6 Rb8 Qe2 Qh1 Ka2 Re8 Qd3 Re1 Qd4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 Bb4 d4 c5 g3 cxd4 Nxd4 O-O Bg2 d5 cxd5 Nxd5 Bd2 Nxc3 bxc3 Be7 Rb1 Nd7 O-O Nb6 Bf4 Nd5 Qb3 Nxf4 gxf4 Qc7 e3 a6 Rfd1 Ra7 c4 Bc5 Nf3 a5 Ng5 h6 Ne4 a4 Qb5 Ra5 Qb2 b6 Qe5 Qxe5 fxe5 Ba6 Bf1 Bb7 Nxc5 Rxc5 f4 Ba6 Rd4 Rfc8 Rb4 a3 Ra4 Ra5 Rb4 Rac5 Be2 g5 Kf2 Kg7 Ke1 R8c6 Rd6 Rc7 Rd4 gxf4 exf4 f6 exf6 Kxf6 Kd2 R7c6 Re4 Kf5 Ke3 Kf6 h3 e5 Bf3 Rd6 h4 Bc8 fxe5 Rxe5 Rxe5 Kxe5 Rb5 Kf6 Bd5 Bd7 Rb3 b5 Rxa3 Be6 Rd3 Bxd5 cxd5 Ke5 h5 Rxd5 Rxd5 Kxd5 Kf4 Kc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 Nc6 Qd2 a6 Nge2 Rb8 Nc1 e5 Nb3 exd4 Nxd4 Ne5 Rd1 c6 Be2 b5 cxb5 axb5 b4 c5 Nc2 cxb4 Nxb4 Be6 O-O Qa5 Qxd6 Nc4 Bxc4 bxc4 Nc6 Qxc3 Bd4 Nxe4 Bxc3 Nxd6 Bxg7 Kxg7 Nxb8 Nf5 Nd7 Rc8 Nb6 Rc6 Rb1 c3 Rb4 Bxa2 Rc1 h5 Kf2 c2 Ke2 Be6 Kd2 Rd6 Kxc2 Ne3 Kb2 Nxg2 Nc4 Rd3 Ne5 Re3 Re4 Kf6 Rxe3 Nxe3 Nd3 Bd5 Ne1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nf6 Bg2 Be7 O-O O-O Nc3 Nc6 Bg5 cxd4 Nxd4 h6 Be3 Re8 Qb3 Na5 Qc2 Bg4 Nf5 Bb4 Bd4 Bxc3 Bxc3 Rxe2 Qd1 d4 Nxd4 Rxf2 Qa4 Rxg2 Kxg2 Qd5 Kg1 Nc4 Qb5 a6 Qxd5 Nxd5 Rfe1 h5 Ne2 Nxc3 bxc3 Ne5 Nd4 f6 a4 Rc8 Ra3 Rc7 Rb3 Kf7 Reb1 Bc8 Kf1 Nc4 Ke2 Na5 Ra3 Nc4 Rab3 Na5 Ra3 Nc4 Ra2 Kg6 Kd3 Ne5 Kd2 Nc4 Kc2 Bh3 Kb3 Bd7 Rc1 Be8 Ne6 Rc6 Nf4 Kg5 Re2 Ne5 Re4 Bf7 Kc2 b5 Rd4 Bc4 axb5 axb5 Ra1 Rc7 h4 Kf5 Nxh5 Kg6 Nf4 Kf7 h5 Rb7 Ra8 g5 Nd5 Nf3 Rd1 Rd7 Ne3 Re7 Nxc4 bxc4 Rd6 Kg7 h6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 c6 Bd3 a6 Nge2 b5 O-O Nbd7 Rc1 e5 a3 exd4 Nxd4 Bb7 cxb5 cxb5 Re1 Ne5 Bf1 Re8 Bf2 d5 exd5 Nxd5 Nxd5 Qxd5 a4 Bh6 Ra1 Nc4 axb5 axb5 Rxa8 Rxa8 Qb3 Bc6 Bd3 Nd6 Qxd5 Bxd5 Nxb5 Nxb5 Bxb5 Bg7 b4 Bc3 Rd1 Bb3 Rb1 Ba2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 bxa4 Rxa4 a5 Ra3 Ra6 Nh2 g6 f3 Qd7 Nc4 Qb5 Rc3 Bc8 Be3 Kh7 Qc1 c6 Ng4 Ng8 Bxh6 Bxh6 Nxh6 Nxh6 Nxd6 Qb6 Nxe8 Qxd4 Kh1 Qd8 Rd1 Qxe8 Qg5 Ra7 Rd8 Qe6 f4 Ba6 f5 Qe7 Qd2 Qe5 Qf2 Qe7 Qd4 Ng8 e5 Nd5 fxg6 fxg6 Rxc6 Qxd8 Qxa7 Nde7 Rxa6 Qd1 Qg1 Qd2 Qf1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Qe7 dxe5 dxe5 Nd5 Qd8 Bc5 Nxe4 Be7 Qd7 Bxf8 Kxf8 Qc2 Nc5 Rd1 Nc6 O-O Ne6 Nb6 axb6 Rxd7 Bxd7 Qd2 Be8 b3 e4 Ne1 f5 Bd1 Ne5 Nc2 Rxa2 Qd5 Ke7 Nb4 c6 Qxe6 Kxe6 Nxa2 Nf7 Be2 Nd6 Nb4 Bc3 Nc2 f4 Rd1 h5 f3 e3 g3 g5 Bd3 h4 Kf1 c5 Ke2 b5 cxb5 Nxb5 Bc4 Ke7 Rd5 Bf6 Rxc5 Nc3 Kf1 Bg6 Ne1 Kd6 Ra5 fxg3 hxg3 hxg3 Ng2 b5 Ra6 Ke7 Ra7 Ke8 Ra8 Bd8 Nxe3 bxc4 Nxc4 g4 Kg2 Ne2 Ne5 gxf3 Kxf3 g2 Rxd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 f5 exf5 Nf6 Ne4 Bxd5 Nxf6 Qxf6 Bd2 Qxb2 Bxb4 Bf7 Re6 Qxb4 Rb3 Qxa4 Bc2 Rad8 Rbe3 Qb4 g3 a5 Nh4 d5 Qe2 Qc4 Bd3 Qc1 Kg2 c4 Bc2 Bxe6 Rxe6 Rxe6 Qxe6 Kh8 Ng6 Kh7 Qe2 Qg5 f6 Qxf6 Nxf8 Kg8 Ng6 Qf7 Ne7 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Na6 O-O c6 dxe5 dxe5 Qxd8 Rxd8 Rfd1 Re8 h3 Bf8 Nd2 b6 a3 Nc5 b4 Ne6 Nb3 Ba6 f3 Nh5 Bf2 Red8 Bf1 Nhf4 g3 Nh5 Kg2 f5 Rab1 Rac8 Rxd8 Rxd8 Rd1 Rxd1 Nxd1 fxe4 fxe4 c5 bxc5 Nxc5 Nxc5 Bxc5 Bxc5 bxc5 Nc3 Nf6 Kf3 Bb7 Bd3 Kf8 h4 h6 Bc2 Ke7 Ba4 a6 Ke3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nd7 d4 Bf6 a4 Bb7 axb5 axb5 Rxa8 Qxa8 d5 Na5 Bc2 Nc4 b3 Ncb6 Na3 Ba6 Nh2 c6 dxc6 Qxc6 Bd2 Be7 Ng4 Ra8 Ne3 Nf6 Nf5 Bf8 Bg5 Nbd7 c4 bxc4 bxc4 Bxc4 Nxc4 Qxc4 Bb3 Qc3 Kh2 h6 Bxf6 Nxf6 Re3 Qc7 Rf3 Kh7 Ne3 Qe7 Nd5 Nxd5 Bxd5 Ra7 Qb3 f6 Qb8 g6 Rc3 h5 g4 Kh6 gxh5 Kxh5 Rc8 Bg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Na6 O-O Ng4 Bg5 f6 Bc1 Kh8 h3 Nh6 dxe5 fxe5 Be3 Nf7 Qd2 Nc5 Ng5 Nxg5 Bxg5 Bf6 Be3 Ne6 Bg4 h5 Bxe6 Bxe6 Nd5 Bh4 Rac1 Kh7 Rc3 Rf7 b3 c6 Nb4 Rd7 Rcc1 Bf6 f4 exf4 Bxf4 Qa5 Nd5 Qc5 Kh1 Bxd5 cxd5 Qd4 dxc6 bxc6 Rxc6 Re8 Rc4 Qxd2 Bxd2 Be5 Be3 Bg3 Rf3 h4 Bf2 Bxf2 Rxf2 Rde7 Rf4 g5 Rf6 Rxe4 Rxe4 Rxe4 Rxd6 Re7 Ra6 Kg7 Kg1
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nd7 d4 Bf6 a4 Bb7 Be3 Na5 Bc2 Nc4 Bc1 d5 dxe5 Ndxe5 Nxe5 Nxe5 axb5 axb5 Rxa8 Qxa8 f4 Ng6 e5 Bh4 Rf1 Be7 Nd2 Bc5 Kh2 d4 Qe2 dxc3 bxc3 Rd8 Ne4 Ba3 Bxa3 Bxe4 Qxe4 Qxa3 f5 Ne7 Qh4 f6 Qg3 Kf8 Kh1 Qc5 exf6 gxf6 Bb3 Nd5 Qh4 Kg7 Rd1 c6 Rd4 Qxc3 Rg4 Kh8 Bxd5 Qa1 Kh2 Qe5 Rg3 cxd5 Qg4 Qc7 Qd4 Qd6 Kh1 Re8 Qg4 Qd7 Rd3 Re1 Kh2 Re4 Qg3 Re5 Ra3 Re8 Qf4 Qb7 Kh1 Qb8 Qh4 Qb6 Qb4 d4 Rg3 Qc7 Rd3 Qc1 Kh2 Qf4 Kg1 Qc1 Kh2 Qf4 Kg1 Rc8 Rd1 Rd8 Qxb5 Qe3 Kh1 d3 Qa5 Qd4 Qa1 Qb6 Qa2 Kg7 Qd2 Qc5 Rf1 Rd4 Rf3 Qd6 Re3 Ra4 Re1 h5 Rb1 Qd7 Qd1 Kh6 Qd2 Kg7 Qe3 h4 Qf3 Kh6 Qe3 Kg7 Qf3 d2 Qh5 Qf7 Qxf7 Kxf7 Rd1 Rd4 Kg1 Rd5 Kf2 Rxf5 Ke2 Rg5 Kf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 cxd4 cxd4 Nc6 Rd1 Qa5 Qxa5 Nxa5 Nf3 O-O Be2 Bd7 Bd2 b6 O-O Rfd8 Rc1 Bg4 d5 Nb7 h3 Bxf3 Bxf3 Nc5 Be3 Rac8 Bg4 Rb8 Rc4 h5 Bf3 e6 Rd1 exd5 exd5 Be5 g4 hxg4 hxg4 Nb7 Ra4 Na5 g5 Rbc8 Be2 Bd6 Kg2 Bc5 Bd2 Rxd5 Bf3 Rdd8 Bxa5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 exd4 e5 Ne4 Qxd4 d5 exd6 Nxd6 Nc3 Nc6 Qf4 Nf5 Bb5 Bd6 Qe4 Qe7 Bg5 f6 Bd2 Bd7 O-O-O Qxe4 Nxe4 Be7 g4 a6 Bc4 Nd6 Nxd6 Bxd6 Rde1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 exd4 Nxd4 Re8 f3 c6 Qd2 d5 exd5 cxd5 O-O Nc6 c5 Rxe3 Qxe3 Qf8 Nxc6 bxc6 Kh1 Rb8 Na4 Rb4 b3 Be6 Nb2 Nh5 Nd3 Rh4 Qf2 Qe7 g4 Bd4 Qxd4 Rxh2 Kxh2 Qh4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nd7 d4 Bf6 a4 Bb7 Na3 exd4 cxd4 Na5 Ba2 b4 Nc4 Nxc4 Bxc4 Re8 Qb3 Rxe4 Bxf7 Kh8 Be3 Re7 Bd5 c6 Be6 Nf8 Bg4 a5 Rac1 Ng6 Bh5 Rc8 Bg4 Rb8 Qc2 Rc7 Qf5 Ne7 Qd3 Nd5 Bd2 c5 Be6 Nb6 dxc5 dxc5 Qxd8 Rxd8 Bf4 Re7 Ng5 Bd5 Bxd5 Rxd5 Rxe7 Bxe7 Re1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 O-O Rc1 Qa5 Nf3 e6 d5 exd5 exd5 Re8 Be2 Bf5 O-O Nd7 h3 Nb6 g4 Bd7 c4 Qxd2 Nxd2 Na4 Bf3 Nc3 Rxc3 Bxc3 Ne4 Rxe4 Bxe4 Re8 Bd3 b6 Kg2 f5 gxf5 Bxf5 Bxf5 gxf5 Rd1 Kf7 Rd3 Bf6 Ra3 a5 Rb3 Bd8 Rc3 Bc7 a4 Kf6 Kf1 f4 Bc1 Kf5 Rc2 Rg8 Re2 Be5 Bb2 Bd4 Bxd4 cxd4 Re7 d3 Ke1 Rc8 Kd2 Rxc4 Kxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 O-O-O g3 Re8 Bb2 f6 Bg2 fxe5 O-O h5 Qd2 Nf6 Qa5 Bb7 Ba3 Qe6 Bxf8 Rhxf8 Qxa7 Qg4 Na3 h4 Nc2 h3 Bh1 Ne4 a4 Nc3 Rae1 Ne2 Rxe2 Qxe2 Nb4 d5 cxd5 cxd5 Bxd5 Bxd5 Nxd5 Qc2 Qa6 Kd7 Ne3 Qe4 Rc1 Rb8 Qf1 Rxb3 Qxh3 Kd8 Qh5 Kc8 Qd1 Rxe3 fxe3 Qxe3 Kh1 Qe4 Kg1 Qe3 Kh1 Qe4 Kg1 Rd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 O-O Nf3 Qa5 Rc1 e6 Bh6 Nc6 h4 cxd4 Bxg7 Kxg7 cxd4 Qxd2 Kxd2 Rd8 Ke3 Bd7 Rb1 Rab8 Bd3 Ne7 h5 f6 hxg6 hxg6 Rh2 b6 g4 e5 dxe5 Bxg4 exf6 Kxf6 Nd4 Rb7 f3 Rbd7 Rb4 Be6 Rc2 a5 Ra4 g5 Bb5 Rd6 Be2 Bd7 Rac4 Re8 Rb2 Nd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Nb6 Nd2 Qe6 b3 a5 Bb2 Bb4 a3 Bxd2 Qxd2 d5 cxd5 cxd5 Rc1 O-O Rxc7 Qg6 f3 Bf5 g4 Bb1 Bb5 Rac8 Rxc8 Rxc8 O-O h5 h3 hxg4 hxg4 Bc2 Qd4 Qe6 Rf2 Rc7 Rh2 Nd7 b4 axb4 axb4 Nf8 Bf1 Bb3 Bd3 Bc4 Bf5 Qe7 Qd2 Rc6 Bd4 Ra6 Bb1 Ra3 Rh3 Rb3 Bc2 Qxb4 Qf2 Ng6 e6 Rb1 Bxb1 Qxb1 Kh2 fxe6 Qb2 Qxb2 Bxb2 Nf4 Rh4 Nd3 Bc3 e5 Kg3 d4 Bd2 Bd5 Rh5 Kf7 Ba5 Ke6 Rh8 Nb2 Re8 Kd6 Bb4 Kc6 Rc8 Kd7 Rc5 Ke6 Rc7 g6 Re7 Kf6 Rd7 Ba2 Ra7 Bc4 Ba5 Bd3 f4 exf4 Kxf4 Bc2 Ra6 Kf7 Ke5 Nd3 Kxd4 Nf2 g5 Bf5 Bd2 Ke7 Kd5 Ne4 Ra7 Ke8 Be3 Nc3 Ke5 Kd8 Bb6 Ke8 Rc7 Ne4 Be3 Ng3 Bf4 Nh5 Ra7 Kf8 Bh2 Ng7 Bg1 Nh5 Bc5 Kg8 Kd6 Kf8 Bd4 Bg4 Be5 Bf5 Rh7 Kg8 Rc7 Kf8 Kc6 Kg8 Re7 Kf8 Bd6 Kg8 Re8 Kf7 Re7 Kg8 Be5 Kf8 Ra7 Bg4 Kd6 Bh3 Ra3 Bg4 Re3 Bf5 Kc7 Kf7 Kd8 Bg4 Bb2 Be6 Bc3 Bf5 Re7 Kf8 Be5 Bd3 Ra7 Be4 Rc7 Bb1 Bd6 Kg8 Ke7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 O-O Nf3 Bg4 Ng5 cxd4 cxd4 Nc6 h3 Bd7 Rb1 Rc8 Nf3 Na5 Bd3 Be6 O-O Bc4 Rfd1 b5 Bg5 a6 Rbc1 Bxd3 Rxc8 Qxc8 Qxd3 Re8 Rc1 Qb7 d5 Nc4 Nd2 Nxd2 Bxd2 Rc8 Rc6 Be5 Bc3 Bb8 Qd4 f6 Ba5 Bd6 Qc3 Re8 a3 Kg7 g3 Be5 Qc5 h5 Bc7 Ba1 Bf4 Qd7 Rc7 Qd8 d6 g5 d7 Rf8 Bd2 Be5 Rb7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nd7 d4 Bf6 a4 Bb7 Na3 exd4 cxd4 Nb6 Bf4 bxa4 Bxa4 Nxa4 Qxa4 a5 Bd2 Re8 d5 Nb4 Bxb4 axb4 Qxb4 Rb8 Qc4 Qc8 Nd4 Ba6 Qc3 c5 dxc6 Bxd4 Qxd4 Qxc6 b4 h6 Re3 Re6 f3 Rc8 Rb3 Bb5 Rb2 Qb7 Nc2 Qe7 Qf2 Rg6 Ne3 Qe5 Rbb1 Bd7 Ra5 Qe7 Ra7 Qd8 Nd5 Kh7 Kh2 Rb8 f4 Re6 Qd4 Qe8 Re1 Bc6 Qd3 Qf8 Rc1 Bxd5 exd5 Rg6 Qf5 Kg8 Rac7 Rf6 Qd7 Rd8 Qxd8 Qxd8 Rc8 Qf8 R1c4 Rf5 Rxf8 Kxf8 Rd4 h5 b5 Ke7 b6 Kd7 g4 hxg4 hxg4 Rf6 Rc4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 c6 d5 Ng4 Bg5 f6 Bh4 Na6 Nd2 Nh6 a3 Nf7 f3 Bh6 Bf2 f5 Qc2 Bd7 b4 c5 Rb1 b6 Nf1 Bf4 g3 Bh6 h4 Nc7 g4 fxg4 fxg4 Bf4 Ne3 Ne8 Ncd1 h6 h5 g5 Rg1 Nf6 Rg2 Qc8 Kf1 Nd8 Kg1 Nb7 Kh1 cxb4 axb4 a5 Nf5 Bxf5 exf5 axb4 Rxb4 Nc5 Rxb6 Nce4 Rc6 Qb7 Be1 Ra1 Bf3 Nc5 Bc3 Rc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 f5 Rae3 Nf6 Nh2 Kh8 b3 bxa4 bxa4 c4 Bb2 fxe4 Nxe4 Nfxd5 Rg3 Re6 Ng4 Qe8 Nxh6 c3 Nf5 cxb2 Qg4 Bc8 Qh4 Rh6 Nxh6 gxh6 Kh2 Qe5 Ng5 Qf6 Re8 Bf5 Qxh6 Qxh6 Nf7 Kh7 Bxf5 Qg6 Bxg6 Kg7 Rxa8 Be7 Rb8 a5 Be4 Kxf7 Bxd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5 Qd2 f5 O-O-O a6 Bd3 c5 dxc6 Nxc6 Nd5 Be6 Bb6 Qd7 Ne2 Rac8 Kb1 Qf7 Rhe1 Kh8 Bc2 Nf6 Bd3 Nd7 Bg1 Nc5 Nb6 Rcd8 Nc3 Nd4 Ncd5 Bxd5 Nxd5 fxe4 fxe4 b5 Rf1 Qd7 cxb5 axb5 Rxf8 Rxf8 h3 Qd8 Bxd4 exd4 Qe2 Qh4 Rf1 Re8 Rf4 Qg5 a3 h5 Ka2 b4 axb4 Ra8 Kb1 Nb3 Kc2 Na1 Kb1 Nb3 Qf2 Qd8 Rf7 Qe8 b5 Ra1 Kc2 Nc5 Rxg7 Kxg7 Qxd4 Qe5 Qxe5 dxe5 b6 Rg1 Ne3 Re1 Nc4 Rg1 Ne3 Re1 Nc4 Rg1 b4 Rxg2 Kc3 Na4 Kb3 Nxb6 Nxb6 Rg3 Kc3 Rxh3 b5 h4 Nc4 Rxd3 Kxd3 h3 b6 h2 b7 h1=Q b8=Q Qf1 Kc3 Qc1 Kb3 Qd1 Ka2 Qa4 Na3 Qxe4 Qc7 Kh6 Nc4 Qd5 Kb2 e4 Qf4 Kg7 Kc3 Qd3 Kb4 Qd4 Qh4 Kf7 Kb5 Qd5 Kb4 Qd4 Qh7 Qg7 Qh1 Qd4 Qh4 Kg8 Qf4 Kg7 Qc1 Kf6 Kb5 Qd5 Kb4 Qd4 Kb5 Qd5 Kb6 Qd4 Kc6 Ke6 Ne3 Qa4 Kb6 Qb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 f5 exf5 Bxd5 Ne4 Bf7 axb5 d5 Nc3 Rxe1 Nxe1 d4 Na2 Nxa2 Bxa2 c4 Rxa6 Nc5 Rxa8 Qxa8 Bb1 d3 Be3 Qa5 b3 Nxb3 Nxd3 cxd3 Bxd3 Nc5 Bf1 Qc7 Qg4 Kh7 Bc4 Bxc4 Qxc4 Qe5 Qf7 Bd6 g3 Qe7 Qg6 Kh8 Bd4 Be5 Bxc5 Qxc5 Qe8 Kh7 Qg6 Kh8 Qe8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5 Qd2 Qh4 g3 Qe7 O-O-O f5 exf5 gxf5 Nh3 Na6 Rg1 Nf6 Nf2 Kh8 Be2 Bd7 Bg5 Nc5 g4 e4 fxe4 fxe4 Be3 Na4 g5 Nxc3 bxc3 Ng8 Ng4 c5 dxc6 Bxc6 h4 d5 cxd5 Bxd5 Qxd5 Rac8 Qd6 Rxc3 Kb1 Qf7 Bd4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 b4 d6 Bb2 Bb7 g3 c5 Bg2 Nbd7 O-O Rc8 d3 Re8 e4 a6 Qb3 b5 Nd2 Rb8 Rfc1 Ba8 Qd1 Qe7 cxb5 axb5 Nb3 e5 bxc5 dxc5 f3 h5 a4 h4 g4 c4 dxc4 bxa4 Ba3 Qd8 Nc5 Bc6 Nxa4 Nh7 Nc5 Ng5 Nxd7 Bxd7 Rc3 Qa5 Rd3 Ba4 Qe1 Qa6 Bc1 Ne6 Rda3 Nc5 Be3 Qd6 Rxa4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f3 O-O Be3 Nbd7 Bd3 c5 Nge2 cxd4 Nxd4 e6 O-O d5 exd5 exd5 Bf2 Ne5 c5 Nfd7 Be2 Nxc5 Ndb5 b6 Nxd5 Bf5 Nd4 Bd3 Bxd3 Qxd5 Bc2 Rad8 Qe2 Rfe8 Rfe1 Nxf3 Qxf3 Rxe1 Rxe1 Bxd4 Qxd5 Rxd5 Re2 Bxb2 Bxc5 Rxc5 g3 b5 Kg2 Be5 Bb3 a5 Rf2 Rc7 Re2 Bc3 Re8 Kg7 Rb8 a4 Bd5 Rc5 Be4 b4 Bc2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 Nd2 Bxc3 bxc3 h6 Bh4 g5 Bg3 d6 e3 Qe7 h4 Rg8 hxg5 hxg5 Be2 Nc6 Qc2 O-O-O O-O-O Rh8 e4 Kb8 Nb3 Nd7 Bd3 Qf6 Rxh8 Qxh8 c5 dxc5 Bb5 Qf8 d5 Nce5 d6 f6 dxc7 Kxc7 Qe2 Qe7 Nd2 Nf8 Nc4 Rxd1 Qxd1 Ng6 Qh5 Bxe4 Qe2 Bd5 Nxe5 Nxe5 Bxe5 fxe5 Qxe5 Kb7 f3 a6 Bd3 b5 Kb2 c4 Bc2 b4 Be4 bxc3 Kc2 Qd7 Kxc3 Qb5 Qxe6 Qa5 Kd4 Qd2 Kc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f3 O-O Be3 Nbd7 Nh3 c6 Qd2 e5 d5 cxd5 cxd5 a6 Nf2 Nh5 Nd3 f5 Be2 fxe4 fxe4 Qh4 Bf2 Qe7 Bxh5 gxh5 O-O-O Qf6 Kb1 Qg6 Bh4 Nf6 Bxf6 Bxf6 Rdf1 Bd7 Rf3 Bg4 Rf2 Bh4 Rff1 Bg5 Qe1 h4 h3 Bh5 Rhg1 Kh8 Rxf8 Rxf8 Rf1 Rxf1 Qxf1 Be7 Ne1 Qg3 Nf3 b5 a3 Bg6 Ka2 Qf4 Qe2 Bd8 Qd2 Qxd2 Nxd2 Ba5 Kb3 Kg7 Kc2 Kf6 Kd3 Bh5 Nf3 Bxf3 gxf3 Bxc3 Kxc3 Kg5 Kd3 Kf4 Ke2 Kg3 b3 a5 a4 bxa4 bxa4 Kg2 Ke3 Kg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 g3 b5 Bg2 Bb7 Bg5 Be7 O-O Na6 Nbd2 c5 c3 cxd4 cxd4 h6 Bxf6 Bxf6 a4 b4 Nb3 O-O Qd3 d5 Rfc1 Qb6 Qb5 Rac8 e3 Nb8 Bf1 Rfd8 h4 g6 Nc5 Bc6 Qd3 Ba8 a5 Qd6 Qb5 Kg7 Qxb4 Nc6 Qd2 e5 b4 Re8 Ra3 exd4 Nxd4 Ne5 Rac3 Rcd8 Bg2 Kg8 Rd1 Nc4 Qc2 Qe7 h5 gxh5 Nf5 Qe5 Nxh6 Kg7 Nf5 Kh8 Rd4 Nd6 Nxd6 Qxd6 Rf4 Bxc3 Qxc3 Kg8 Bf3 Re5 Qd4 Qe7 Rh4 Qf6 Rxh5
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 Nc6 Bg2 d6 Nf3 e5 d3 f5 O-O Nf6 Rb1 h6 b4 O-O b5 Ne7 a4 Be6 Ba3 Rc8 Nd2 b6 e3 g5 d4 exd4 exd4 f4 Re1 Bg4 Nf3 Qd7 c5 Rce8 Rc1 Nf5 Qd3 Kh8 cxd6 cxd6 Rxe8 Qxe8 Rf1 Qh5 Ne4 Nxe4 Qxe4 Bh3 Ne5 Bxg2 Kxg2 g4 Bxd6 Rf6 Bb8 Qh3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 d5 a3 Bxc3 Nxc3 cxd4 exd4 dxc4 Bxc4 Nc6 Be3 O-O O-O b6 Qf3 Bb7 Bd3 Rc8 Rad1 Rc7 Qh3 Ne7 Bg5 Ng6 Qg3 Rd7 d5 exd5 Bf5 Re7 h4 Qc7 Qxc7 Rxc7 Rfe1 Re7 Nb5 Rfe8 Rxe7 Nxe7 Bh3 Bc8 Bxc8 Rxc8 Nxa7 Rc2 b4 Kf8 Be3 Nf5 Bxb6 Ng4 Bc5 Ke8 Nb5 Ra2 Nc3 Rxa3 Nxd5 f6 b5 Rb3 b6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 Rb1 a5 a3 Nge7 e3 d6 b4 axb4 axb4 O-O Nge2 Be6 b5 Na5 d3 c6 O-O d5 cxd5 cxd5 Ba3 Re8 Bb4 b6 Ra1 Qd7 Bxa5 Rxa5 Rxa5 bxa5 Qa4 d4 exd4 exd4 Ne4 Rb8 Rb1 Qd8 Nf4 Bf5 Qa2 Bxe4 Bxe4 Be5 Nd5 Nxd5 Bxd5 Qc7 Bc4 Bd6 Re1 Kg7 Re4 Be5 h4 h5 Qe2 Bf6 Qd1 Qc5 Rf4 Rb7 Qf3 Rb6 Qa8 Qc7 Qxa5 Rb7 Qa6 Rb6 Qa8 Rb7 Qe8 Qe7 Qc6 Rc7 Qf3 Qd6 Qe4 Re7 Qa8 Qb6 Kg2 Qd6 Re4 Rxe4 Qxe4 Bd8 Qb7 Bc7 Qc8 Qe7 Bd5 Qd6 Be4 Qc5 Qd7 Qd6 Qe8 Qd8 Qc6 Qd6 Qc4 Bb6 Bd5 Qc7 Qa2 Qd7 Bc4 Qc7 Qa8 Qa7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Be6 O-O Be7 d4 Nxc3 bxc3 e4 Nd2 f5 e3 O-O c4 Nb4 Rb1 b6 Bb2 a5 Ba1 Nxa2 f3 exf3 Qxf3 Bb4 Rf2 Qe7 Qd1 Nc3 Bxc3 Bxc3 Bxa8 Rxa8 Nf3 Bb4 Qd3 Re8 Ne5 Bd6 Rb5 Bc8 Qb3 Qe6 Qa4 Bb7 Qa2 Ba6 Rd5 Bb7 Rb5 Ba6 Rd5 Bb7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 f5 Nf3 Nf6 g3 e6 Bg2 Be7 O-O O-O Nc3 d5 cxd5 exd5 e3 Kh8 b3 Nc6 Bb2 Be6 Ne2 Bg8 d3 Qd7 a3 Bd6 Qc2 Rae8 b4 Ng4 Ned4 Nce5 Rac1 c6 Bc3 Nxf3 Nxf3 Re7 Qb2 Qe8 Bd4 a6 Bc5 Rf6 Bxd6 Rxd6 Qd4 Bf7 a4 Bh5 Rb1 Nh6 b5 axb5 axb5 Bxf3 Bxf3 Nf7 Bh5 Rh6 Bxf7 Rxf7 bxc6 bxc6 Rb6 Rf8 Qc5 g5 Rc1 f4 exf4 gxf4 Rxc6 Rxc6 Qxc6 Qxc6 Rxc6 Ra8 Rf6 fxg3 hxg3 Ra3 Rf3 Kg7 Kg2 d4 Kh3 h5 Kh4 Kg6 g4 hxg4 Kxg4 Ra2 Kf4 Re2 Rg3 Kf6 Rg2 Ke6 Kf3 Re1 Rg4 Kd5 Rg5 Kd6 Ra5 Ke6 Ra2 Kd5 Kf4 Kd6 Kf5 Kc5 f4 Kb4 Rc2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 Nc3 g6 Be2 Bg7 h4 h5 Nf3 Nc6 d5 Ne5 Nxe5 dxe5 Bg5 a6 Qd2 b5 f3 Rb8 a4 b4 Nd1 c6 dxc6 Qc7 Ne3 Qxc6 Nc4 Nd7 a5 Qe6 Qd5 Qxd5 exd5 Bh6 Kf2 f6 Bxh6 Rxh6 Rhd1 Rh8 g3 Kf7 f4 exf4 gxf4 Bb7 Rd4 Ba8 Ne3 Nc5 Rg1 b3 c3 Rbd8 Rc4 Nb7 Rc7 Nxa5 f5 g5 hxg5 fxg5 Rxg5 Nb7 Bxh5 Kf8 f6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 b3 Bg7 Bb2 d6 d4 O-O e3 Nbd7 Be2 e5 O-O e4 Nfd2 Re8 Nc3 Nf8 b4 h5 b5 N8h7 Rc1 Bf5 a4 c6 Ba3 h4 Rb1 Ng5 Kh1 h3 g3 Bf8 Nb3 Qd7 a5 cxb5 Nxb5 Bg4 Nd2 b6 Nc3 bxa5 Rb5 Bf5 Nd5 Nxd5 Rxd5 Nh7 Rxa5 Nf6 Ra6 Reb8 c5 Nd5 Qc2 dxc5 dxc5 Nc7 c6 Qe6 Ra5 Bxa3 Rxa3 Rb6 Rc1 a6 Nb3 Bg4 Nd4 Qe7 Raa1 Bxe2 Qxe2 Ne6 Qg4 Nxd4 exd4 Qe6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 b3 Bg7 Bb2 c5 g3 d6 d4 Ne4 Qc1 Qa5 Nbd2 Nxd2 Qxd2 Qxd2 Kxd2 Nc6 e3 Bg4 Be2 Rb8 Kc1 cxd4 Nxd4 h5 f3 Bd7 Nxc6 Bxb2 Kxb2 bxc6 Kc3 c5 h4 a5 a3 Bc6 Rhb1 Kd7 b4 axb4 axb4 cxb4 Rxb4 Rxb4 Kxb4 Rb8 Kc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 f4 d5 exd5 Nf6 Bb5 Bd7 Bxd7 Qxd7 c4 e6 Qe2 Bd6 d3 O-O dxe6 fxe6 Nf3 Nc6 O-O Rae8 Nc3 e5 f5 Qxf5 Bg5 e4 dxe4 Nxe4 Rae1 Nf6 Qd1 Rxe1 Rxe1 Nd4 Bxf6 Nxf3 Qxf3 Qxf3 gxf3 Rxf6 Re8 Rf8 Re6 Rd8 Kg2 Kf7 Re4 g6 b3 Bf8 Nd5 b5 Rf4 Kg7 Kh3 bxc4 bxc4 Rb8 Rf6 Rb7 Ra6 Kf7 f4 Bg7 Kg4 Bd4 h4 Kg7 h5 Rb2 Rxa7 Kh6 hxg6 hxg6 Ne7 Kg7 Nf5 Kf6 Ra6 Kf7 Nd6 Ke7 a4 Ra2 Kf3 Ke6 Ne4 Ke7 a5 Ra4 Rxg6 Rxa5 Nd6 Bf6 Ke4 Ra6 Nb5 Kf7 Kf5 Rc6 Nc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 Nd2 Qe8 b3 Nh7 a3 f5 f3 h5 Rb1 g5 Bf2 g4 exf5 Bxf5 Nde4 Qg6 h3 gxh3 gxh3 Kh8 Rg1 Qh6 h4 Qf4 Qd2 Bh6 Qxf4 Bxf4 b4 axb4 axb4 Nf6 Bd3 Nxe4 Nxe4 Nb8 Rb2 Nd7 Ke2 Ra3 c5 Nf6 cxd6 cxd6 Rd1 Nxd5 Bc4 Bxe4 Bxd5 Bxd5 Rxd5 Bc1 Rc2 Raxf3 Bb6 Ba3 Rxd6 Bxb4 Rh6 Kg8 Rxh5 Rg3 Rc7 Rf7 Rc8 Rf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 f3 d5 e5 Nfd7 c4 dxc4 Bxc4 e6 f4 c5 dxc5 Bxc5 Nf3 a6 Nc3 b5 Bd3 Bb7 a3 Nc6 Ne4 O-O h4 Be7 Be3 f6 Bc2 fxe5 Nfg5 Qe8 Qd3 Rf5 O-O-O Nf8 Nc5 Bxc5 Bxc5 Rc8 Kb1 Nd4 Bxd4 exd4 Rhe1 Rfc5 Rxe6 Qd7 Bb3 Bd5 Rxa6 Qc7 Ne4 Rc1 Ka2 Bc4
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 d4 Nf6 e4 d6 Nf3 O-O h3 e5 d5 Na6 Be3 Nh5 Nh2 Qe8 Be2 f5 exf5 Nf4 O-O Bxf5 Re1 Qf7 Nf1 Bxh3 gxh3 Nxh3 Kg2 Nxf2 Qb1 e4 Ng3 Qd7 Bxf2 Rxf2 Kxf2 Qh3 Qxe4 Nc5 Qe7 Be5 Rg1 Rf8 Ke1 Bxg3 Kd2 Rf7 Qe8 Rf8 Qe7 Bf4 Kc2 Qf5 Kd1 Rf7 Qe8 Kg7 Rf1 Qf6 Kc2 Rf8 Rxf4 Qxf4 Qe7 Rf7 Qe8 Qg5 Rd1 Re7 Qc8 Qf6 Bd3 Qf2 Rd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qc7 Qf3 Nbd7 O-O-O b5 e5 Bb7 Qh3 dxe5 Nxe6 fxe6 Qxe6 Be7 Bxf6 gxf6 Be2 h5 Nd5 Bxd5 Rxd5 Nc5 Qf5 Qc6 Qg6 Kf8 Rhd1 Qe8 Qf5 Qc8 Qg6 Qe6 Bxh5 Qg8 Rd8 Rxd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 f5 d4 Nf6 g3 d6 Bg2 g6 Nc3 Bg7 d5 O-O Nh3 c6 Nf4 e5 dxe6 Qe7 O-O Bxe6 Nxe6 Qxe6 Bf4 Qxc4 Qxd6 Na6 Rfd1 Nh5 Bg5 Bf6 Bxf6 Nxf6 Rac1 Qb4 Qxb4 Nxb4 a3 Na6 e4 fxe4 Nxe4 Nxe4 Bxe4 Rad8 h4 Nc7 Bb1 Kg7 Ba2 Kf6 Kg2 Nb5 g4 h5 g5 Kg7 Rxd8 Rxd8 Re1 Rd7 Re8 Nd6 Rg8 Kh7 Rb8 Kg7 Rg8 Kh7 Rb8 Kg7 Bb1 Nc4 Re8 Nxb2 Re6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 Be2 e6 Bf3 Nc6 Be3 Be7 O-O O-O Kh1 Bd7 g4 Nxd4 Bxd4 Bc6 g5 Nd7 Bg2 b5 a3 a5 Qe2 Rab8 b4 Rfe8 Rad1 Bf8 Rd3 e5 Be3 exf4 Bxf4 Ne5 Rh3 Bd7 Nd5 Qc4 Qxc4 Nxc4 Rd3 a4 h4 Rbc8 h5 Bg4 Bf3 Bxf3 Rdxf3 Rxe4 Nc3 d5 Nxb5 d4 Rd1 Ne5 Rff1 Rxc2 Nxd4 Rc4 Bxe5 Rxe5 Nf3 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 e6 O-O Nge7 c3 d5 exd5 Qxd5 Re1 Bd7 Na3 Ng6 Bf1 Be7 d4 cxd4 Nb5 Rc8 Nfxd4 Nxd4 Nxd4 a6 g3 b5 Bg2 Qc5 h4 Bf6 Be3 Qc7 Nf5 O-O Nd6 Rcd8 Nb7 Rc8 Nd6 Rcd8 Nb7 Rc8 h5 Ne5 Bf4 Bc6 Bxc6 Qxc6 Bxe5 Bxe5 Rxe5 Qxb7 h6 Qe7 hxg7 Rfd8 Qe2 Rc5 a4 Rxe5 Qxe5 bxa4 Qe4 Qc5 Rxa4 a5 Qe2 Qg5 Rf4 Rd7 Qf3 Qd5 Qxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 a6 Qd2 Nbd7 Nge2 c5 d5 b5 cxb5 Ne5 Nc1 axb5 Nxb5 Ba6 a4 c4 Nc3 Nfd7 Be2 Rb8 Be3 Qa5 Ra3 Nc5 Nd1 c3 bxc3 Rb1 Bxc5 Qxc5 Rb3 Rxc1 Bxa6 Ra1 Be2 Nc4 Bxc4 Qxc4 Qb2 Rxa4 Kf2 Qc5 Ne3 Bh6 Re1 Rfa8 Rb8 Rxb8 Qxb8 Kg7 Qb2 Qa7 Qc1 Ra3 Kg3 Qc5 Qb2 Bxe3 Rxe3 Qxe3 Qxa3 Qe1 Kh3 h5 Qa6 g5 g3 g4 fxg4 hxg4 Kh4 Qxe4 Qc8 Qh7 Kxg4 Qxh2 Qf5 Qe2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3 f5 Rae3 Nf6 Nh2 Qd7 exf5 Rxe3 fxe3 Bxd5 Ng4 Be7 e4 Bf7 Nf3 Rd8 Qe2 Bc4 Qe3 Nxg4 hxg4 Bf6 Bd2 Re8 b3 Bf7 Qf2 Qe7 axb5 axb5 Qf1 Rb8 Bd3 Nxd3 Qxd3 c4 bxc4 bxc4 Qa3 Rb3 Qa8 Qe8 Qxe8 Bxe8 g5 hxg5 Bxg5 Bb2 Rd1 Ba3 e5 c3 exd6 c2 Rd5 Rb1 Kf2 Rd1 Ra5 c1=Q Bxc1 Bxc1 Ra6 Rd5 f6 g6 g4 g5 Ra7 Bf4 f7 Bxf7 d7 Kf8 Ng1 Rd2 Ke1 Rd5 Nh3 Bd2 Ke2 Kg7 Ra2 Bc1 Rc2 Ba3 Nxg5 Bg8 Rc7 Kg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5 f3 f4 Bf2 g5 a4 Ng6 Nd3 Nf6 c5 h5 h3 Rf7 c6 a5 cxb7 Bxb7 b4 Bc8 bxa5 Bh6 Nb4 g4 Nc6 Qf8 fxg4 hxg4 hxg4 Bg5 Bf3 Qh6 Re1 Nh4 Bxh4 Bxh4 g5 Qxg5 Re2 Ng4 Rb1 Bg3 Qd3 Qh4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 Nc3 Bb7 d4 e6 a3 d5 cxd5 Nxd5 e3 g6 Bb5 c6 Ba4 Bg7 e4 Nxc3 bxc3 Ba6 h4 Qc7 e5 Nd7 Bf4 h6 Qc1 O-O-O Qe3 c5 Nd2 Kb8 Bd1 g5 Bg3 gxh4 Rxh4 Nf8 a4 Ng6 a5 Nxh4 Bxh4 cxd4 cxd4 Rd5 axb6 Qxb6 Be7 Bf8 Bxf8 Rxf8 Qa3 Rfd8 Qxa6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 d4 Nf6 O-O a6 Bxd7 Nxd7 Nc3 e6 Bg5 Qc7 Re1 cxd4 Qxd4 Ne5 Rad1 Bd7 Nxe5 dxe5 Qd2 b5 a3 Rc8 Re3 f6 Rd3 Bc6 Be3 Kf7 f3 a5 Qf2 b4 axb4 axb4 Na2 Bb5 R3d2 Ra8 Nc1 Be7 Bb6 Qc6 Nb3 Rhc8 Be3 g5 g4 h5 h3 Kg6 Kg2 h4 Qg1 Ra2 Rb1 Bc4 Nc1 Raa8 Nb3 Rd8 Rbd1 Rxd2 Rxd2 Ra2 Qb1 Qa4 Bc5 Bxc5 Nxc5 Qa7 b3 Ra1 bxc4 Rxb1
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e6 e4 d6 d4 Be7 Nf3 O-O Bd3 c5 d5 e5 h3 Na6 a3 Nc7 g4 Nfe8 Ne2 a6 a4 Rb8 b3 b5 Bd2 bxc4 bxc4 Bd7 a5 Rb7 Ng3 g6 Bc2 Ng7 Rb1 Qb8 Ke2 Kh8 Rxb7 Qxb7 Qb1 Rb8 Qxb7 Rxb7 Rb1 Rxb1 Bxb1 h5 Nh2 h4 Ngf1 g5 Ne3 Kg8 Nf3 f6 Bc2 Kf7 Nd1 Bd8 Nb2 Ke7 Ba4 Bxa4 Nxa4 Na8 Ne1 Kd7 Nc2 Ke7 Ne3 Kd7 Nf1 Ke7 Bc3 Kd7 Nd2 Kc7 Nb3 Be7 Bd2 Ne8 Be3 Kc8 Kd3 Bf8 f3 Be7 Nbxc5 dxc5 Nxc5 Bxc5 Bxc5 Kd7 Bb6 Nxb6 axb6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 f5 Nf3 Nf6 Bg5 c6 e3 Be7 Bd3 Ne4 Bxe7 Qxe7 O-O Nd7 Ne2 O-O Ne1 b6 Rc1 Bb7 Bb1 Rac8 Nd3 Kh8 a3 Ba6 Ba2 Bb7 b4 c5 cxd5 exd5 bxc5 bxc5 dxc5 Ndxc5 Ndf4 Qe5 Qd4 Qe7 Nxd5 Bxd5 Bxd5 Rcd8 Rcd1 Nf6 Qc4 Ncd7 Qa2 Ne5 Nf4 Rd6 Bb3 Ra6 a4 Ne4 Rd4 Rh6 f3 Ng5 Bd5 Nexf3 Bxf3 Qxe3 Qf2
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 d6 g3 g6 Bg2 Bg7 d4 O-O O-O c6 Nc3 Qc7 e4 e5 h3 Nbd7 Be3 Re8 d5 c5 Ne1 a6 a4 b6 Nd3 Rb8 Qd2 Nh5 Bf3 Nhf6 g4 Kh8 Bg2 Nf8 b4 Bd7 Kh1 a5 bxa5 bxa5 f3 Ng8 Nb5 Bxb5 axb5 Ra8 Ra2 Ra7 g5 f5 h4 f4 Bf2 h6 gxh6 Nxh6 Be1 Bf6 Bh3 Qe7 Qh2 Nd7 Bg4 a4 Bxd7 Qxd7 Rg1 Rg8 Rg5 a3 Nf2 Bxg5 hxg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Qc2 Bd6 Be2 O-O O-O b6 e4 dxe4 Nxe4 Nxe4 Qxe4 Bb7 Rd1 Qc7 Bd3 g6 Qh4 Rfe8 Ng5 Nf8 Ne4 Be7 Bg5 Bxg5 Qxg5 Red8 Nf6 Kh8 Be4 Nd7 h4 Qd6 c5 Qf8 Bf3 Nxf6 Qxf6 Qg7 Qf4 Rd7 Rd2 Kg8 Rc1 Rc8 b4 b5 Re1 Rcd8 Re5 Qf8 h5 Qg7 Bd1 Bc8 Bc2 Rb7 h6 Qf8 Bb3 Rbd7 g4 Rb7 g5 Rbd7 Re4 Bb7 Qf6 Re8 Rde2 Bc8 a4 Bb7 Rf4 Rde7 Re5 a6 a5 Bc8 Re3 Bd7 Qe5 Rc8 Bc2 Rce8 Bxg6 hxg6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 Bc4 e6 N1e2 h6 O-O Bd6 Nf4 Bh7 Ngh5 Bf8 Re1 Ne7 Bxe6 fxe6 Nxe6 Qd7 Qf3 Rg8 Nxf8 Rxf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 Nf3 Nf6 Nc3 g6 e4 Bg7 h3 O-O Be3 c6 a4 a5 Nd2 Nbd7 Be2 Qc7 O-O Ne8 Nc4 e5 dxe5 dxe5 Qd2 b6 Rad1 Nef6 f4 Ra7 fxe5 Nxe5 Nxe5 Qxe5 Bd4 Qe6 Qf4 Nd7 Bxg7 Kxg7 Rd6 Qe5 Rxc6 Qd4 Kh1 Nc5 Nb5 Qd7 Nxa7 Qxa7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 c5 e3 b6 Nc3 Be7 d5 exd5 cxd5 d6 e4 O-O Be2 Re8 O-O Na6 Nd2 Nc7 a4 Bf8 f4 Rb8 Kh1 a6 g4 b5 g5 Nd7 Bf3 Nb6 Bg2 bxa4 Nxa4 Nxa4 Rxa4 Nb5 Nc4 Be7 Be3 f6 gxf6 Bxf6 e5 Be7 Qd3 Bf8 Be4 Qh4 e6 Be7 Rg1 Bf6 Rg3 Rf8 Qf1 g6 Qg2 Bg7 Ra1 Qe7 Rg1 Nd4 Bxg6 hxg6 Bxd4 cxd4 Rxg6 Rb7 Nxd6 Ra7 Nxc8
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d3 Nf6 d4 d5 Bd3 Be7 O-O Bg4 Nbd2 O-O Re1 Nc6 c3 Re8 Qb3 Rb8 Ne5 Bd6 f4 Na5 Qc2 c5 h3 Bc8 Ndf3 c4 Bf5 Bxf5 Qxf5 Qc8 Nh4 Nc6 Be3 Ne7 Qxc8 Rbxc8 g4 Ne4 Ng2 f6 Nf3 b5 a3 a5 Nd2 Nc6 Nxe4 Rxe4 Bd2 Rce8 Rxe4 Rxe4 Re1 Rxe1 Bxe1 g6 Kf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 d6 e4 c5 dxc5 dxc5 c3 O-O Bc4 a6 a4 Nc6 O-O Nd7 Qc2 Nde5 Nxe5 Nxe5 Be2 h6 Bh4 g5 Bg3 Be6 Rfd1 Qc7 Bc4 Qc6 Bxe6 Qxe6 Nf1 Qc6 Ne3 e6 f3 Rfd8 Rxd8 Rxd8 Rd1 Rxd1 Qxd1 Qd7 Qxd7 Nxd7 a5 Ne5 Bxe5 Bxe5 Nc4 Bc7 e5 Kf8 Kf2 Ke7 Ke3 f6 exf6 Kxf6 h3 h5 g4 h4 Ke4 Bf4 b4 cxb4 cxb4 Bc7 Nb2 b6 axb6 Bxb6 Kd3 Ke7 Nc4 Bc7 Nd2 Kf7 Ne4 Kg6 Kc4 a5 b5 Bb6 Kb3 Kh6 Ka4 Kg6 Nd6 Kf6 Nc4 Bd8 Nxa5 Bc7 Nc4 e5 Kb4 Ke6 Kc5 Bd8 Kc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 Bg4 Be3 Nbd7 Nd2 Bxe2 Qxe2 c5 d5 Rb8 O-O Ne5 h3 Ne8 f4 Nd7 Nf3 Nc7 e5 a6 a4 b6 Bf2 Re8 Rae1 Nf8 Bh4 Qd7 Ng5 h6 e6 fxe6 Nxe6 Ncxe6 dxe6 Qc6 Nd5 Rb7 f5 Be5 fxg6 Nxe6 Qh5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Nbd2 Nc5 c3 d4 Bxe6 Nxe6 cxd4 Ncxd4 Ne4 Be7 Be3 Nf5 Qc2 O-O Neg5 Nxg5 Bxg5 Bxg5 Qxf5 Bh6 Rfd1 Qe7 Rd7 Qc5 Rad1 Rae8 g3 f6 R7d5 fxe5 Qe4 Qb6 Kg2 Qf6 Rd7 Qf5 Qd5 Kh8 Rxc7 e4 Qxf5 Rxf5 Ne1 Ref8 Rc2 g6 Re2 Bg7 b3 a5 Nc2 a4 Ne3 Rc5 Rd5 Rxd5 Nxd5 axb3 axb3 Bd4 Nc7 Rf5 Ne6 Bb6 Nf4 g5 Nh5 Kg8 g4 Rf3 b4 Rb3 Nf6 Kg7 Nxe4 Rxb4 h3 h6 Nd6 Rb3 Re6 Bc5 Nf5 Kf7 Rc6 Bf8 Nxh6 Bxh6 Rxh6 Rb1 h4 gxh4 Rxh4 b4 Rh5 b3 Rb5 b2 Kf3 Kf6 Kf4 Kg6 Rb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 e6 e4 Nb6 Nc3 Be7 Bd3 O-O O-O c5 dxc5 Bxc5 Qe2 Nc6 e5 Nd4 Nxd4 Bxd4 Rd1 f5 exf6 Qxf6 Be3 e5 Ne4 Qh4 Bxd4 exd4 Bc4 Kh8 Rxd4 Nxc4 Qxc4 Bf5 Ng3 Qg5 Nxf5 Rac8 Qd3 Rxf5 Rd1 Rcf8 Qd2 Qf6 Rd8 Qb6 Rxf8 Rxf8 Qd4 Qc7 Qxa7 Qc2 Qd4 h6 Qd2 Rc8 h3 Qf5 Qd3 Qf4 g3 Qf6 Qb3 Qf5 h4 Qe4 Qd5 Qe2 b4 b5 a3 Rc3 Qd2 Qf3 Re1 Rxa3 Re8 Kh7 Qc2 Rd3 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 g3 d5 Bg2 Nf6 Nf3 Be7 d4 O-O O-O c6 Qc2 b6 Rd1 Bb7 b3 Nbd7 Nc3 Rc8 e4 Nxe4 Nxe4 dxe4 Qxe4 Nf6 Qe2 Qd6 Ne5 Qb8 Bf4 Qa8 Rd3 c5 Bxb7 Qxb7 dxc5 Bxc5 Rad1 Qa8 Bg5 Ne4 Be3 Bxe3 Qxe3 Nf6 g4 h6 h4 Rcd8 f3 Rxd3 Rxd3 Rd8 b4 Rxd3 Qxd3 Qb8 Qd4 Qc7 Kg2 Kf8 b5 Kg8 Nc6 Nd7 f4 Kf8 Kf3 Kg8 g5 hxg5 hxg5 a5 Ke3 Kf8 Qe4 Qd6 Qd3 Qc5 Kf3 Ke8 Qh7 Qa3 Kg4 f5 Kh4 Qb2 Qh5 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 bxc6 O-O Nf6 Nxe5 Bd6 Nf3 Be7 e5 Nd5 c4 Nf4 d4 Ng6 Nc3 d5 exd6 cxd6 h3 Bf5 Re1 O-O b3 d5 c5 Qd7 Bg5 f6 Be3 Bd8 Na4 Rb8 Nd2 Bc7 Nb6 Qf7 Nf1 Nh4 Ng3 Qg6 Qh5 Bxg3 Qxg6 Bxf2 Bxf2 Nxg6 g4 Be4 Nd7 Rb4 Nxf8 Nxf8 Rxe4 dxe4 Re1 Ne6 Rxe4 Kf7 Kf1 Nxc5 Re2 Ne6 Rc2 Nxd4 Bxd4 Rxd4 Rxc6 Rd3 Rxa6 Rxh3 Rb6 Re3 a4 Rc3 a5 Kg6 a6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 f5 d3 d6 e4 e5 g3 Nf6 Bg2 Be7 O-O O-O Nbd2 Nc6 c3 Kh8 Re1 f4 d4 fxg3 hxg3 exd4 cxd4 Ng4 Nc4 Qe8 Ne3 Qh5 Nd5 Rxf3 Qxf3 Bd7 Qd3 Nxd4 Nxc7 Rc8 Qxd4 Rxc7 a4 Rc2 Bd2 Be6 Rac1 Qh2 Kf1 Rxc1 Rxc1 Ne5 f4 Bh3 Qg1 Qxg3 Rc3 Nd3 Ke2 Bg4 Kf1 Bh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 h6 Bh4 Nbd7 Nge2 Re8 Qc2 c5 dxc5 Nxc5 O-O Nxd3 Qxd3 Be6 Nf4 Qa5 Rfd1 Rad8 h3 Rd7 a3 Red8 b4 Qc7 Rac1 Qe5 Nfe2 Qh5 Bg3 Ne4 Nf4 Qf5 Nxe6 Qxe6 Ne2 Nxg3 Nxg3 Bf6 Ne2 Qe4 Kf1 Qe5 Kg1 Qe4 Rc2 Qxd3 Rxd3 Kf8 f4 Ke7 Kf2 g6 g4 Bg7 Kf3 f5 gxf5 gxf5 Ng3 Ke6 b5 a6 bxa6 bxa6 Rb3 Bf8 a4 a5 Rb5 Bb4 Ne2 Kf6 Nd4 Rg8 Rc6 Rd6 Rxd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e6 Bxc4 Nf6 e5 Bb4 Nc3 Nd5 Ne2 O-O O-O c6 Ne4 Nd7 N2g3 b5 Bd3 Be7 Nh5 f5 exf6 N7xf6 Nhxf6 Nxf6 Be3 Qd5 Nxf6 Bxf6 Qc2 Bb7 Bxh7 Kh8 Be4 Qd7 Qe2 Rfd8 Qh5 Kg8 Rac1 Rac8 Rfd1 a5 Rc5 Kf8 g4 Qf7 Bg6 Qd7 g5 Ke7 gxf6 gxf6 Qh7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Bd3 O-O Qc2 Nbd7 Nge2 Re8 O-O Nf8 f3 Nh5 Bxe7 Rxe7 e4 dxe4 fxe4 Qd6 Qd2 Bg4 e5 Qd7 h3 Bxe2 Nxe2 g6 Rad1 Rd8 Qh6 Ne6 Bc4 Qe8 g4 Nhg7 Ng3 Qf8 Ne4 Ne8 Qe3 N8c7 Rd2 Kg7 h4 h6 h5 g5 Rf6 Nd5 Bxd5 Rxd5 Rdf2 Qc8 Qf3 Nf4 Rxf4 gxf4 Qxf4 Qe6 g5 Rxd4 Qf6 Kg8 gxh6 Rd1 Kh2 Qxe5 Qxe5 Rxe5 Nf6 Kh8 Rg2 Rd8 h7 Ree8 Rg4 c5 Kg3 b5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 e3 g6 Nc3 Bg7 b4 Bg4 Bb2 O-O Qb3 Bxf3 gxf3 e6 f4 Nbd7 Be2 a6 c5 Ne8 Bf3 f5 a4 Nc7 Ke2 Qe7 b5 Rfb8 Ba3 Qd8 Rhb1 Qc8 Qc2 Bf6 Kf1 Kf7 Be2 h6 Bb4 axb5 axb5 Rxa1 Rxa1 Ra8 Ra2 Ne8 Ra4 Bd8 Qa2 Rxa4 Qxa4 Nb8 Ba5 Be7 Bb6 Nf6 Qa8 Nfd7 Ba7 Ke8 Na2 Kd8 Nb4 Kc7 Bxb8 Nxb8 b6 Kd8 Nd3 Nd7 Qa7 Bf6 Nc1 Qb8 Nb3 Kc8 Na5 Be7 Ba6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 cxb5 a6 f3 g6 e4 d6 Na3 Bg7 Ne2 O-O Nc3 axb5 Naxb5 Bd7 a4 Na6 Bc4 Nb4 O-O Ne8 Be3 Nc7 Qd2 Nxb5 Nxb5 Re8 f4 Qb8 f5 Na6 Bh6 Nc7 Bxg7 Kxg7 Nc3 Qb4 b3 f6 Qe3 Rh8 Rad1 Be8 e5 Bf7 exf6 exf6 fxg6 Bxg6 Qe7 Kh6 Rf3 Ne8 Rh3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 e3 Nf6 Nf3 Nc6 cxd5 exd5 Bb5 cxd4 exd4 Bb4 O-O O-O Bxc6 bxc6 Ne5 Bxc3 bxc3 Ne4 Ba3 Re8 Re1 Be6 c4 Qa5 Re3 Qd2 Qxd2 Nxd2 cxd5 Bxd5 Rc1 Rac8 Bb4 Ne4 f3 Nf6 Ra3 Rb8 Bc3 Rb7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 h6 b4 Bb6 a4 a6 O-O d5 exd5 Nxd5 Re1 Bg4 h3 Bxf3 Qxf3 Nf6 Nd2 O-O Ba3 Re8 Ne4 Rc8 Rad1 Nb8 Bc1 Nbd7 Ng3 c6 Nf5 Qc7 Bxh6 g6 Qg3 Nd5 Bd2 e4 Qxc7 Rxc7 Ne3 N5f6 d4 Ba7 a5 Rcc8 Bf1 Kg7 g3 Rh8 Bg2 Nh7 Nc2 Ndf6 c4 Rcd8 Bc3 Ng5 h4 Ngh7 Rd2 Nf8 Ne3 Rd7 d5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Bd6 O-O O-O c4 c6 Qc2 Re8 Nc3 Bg4 Ne5 Qh4 g3 Qh5 Nxg4 Nxc3 bxc3 Qxg4 Bxh7 Kh8 Bd3 g6 Rb1 Kg7 Rxb7 Rh8 c5 Qc8 Rb2 Be7 f4 Bf6 Qf2 Nd7 Re1 Qc7 h4 Rae8 Rbe2 Rxe2 Rxe2 Qa5 Qe1 Qd8 Kh2 Kf8 Be3 Be7 Rb2 Qa5 Bf2 Nf6 Rb7 Bd8 f5 gxf5 Bxf5 Qxa2 Rb8 Kg7 Qe3 Nh7 Bxh7 Kxh7 Qf4 Bf6 Qf5 Kg7 Rxh8 Kxh8 Qxf6
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 g3 Nc6 Bg2 g6 Nc3 Bg7 e3 d6 Nge2 f5 d3 Nf6 Rb1 O-O b4 Qe8 b5 Ne7 a4 g5 f4 gxf4 exf4 c6 O-O Kh8 Qd2 Ng6 Bb2 exf4 Nxf4 Bd7 Nce2 Ne5 Rbe1 Rd8 Kh1 Qf7 c5 cxb5 cxd6 Ne8 Ng1 Ng6 Nh5 Bxb2 Qxb2 Kg8 Nf3 Bc6 axb5 Bxf3 Bxf3 Rxd6 Qa3 Qd7 Qxa7 Qxb5 Bxb7 Qxd3 Bg2 Rd7 Qa2 Kh8 Bc6 Re7 Qa1 Kg8 Rxe7 Nxe7 Bxe8 Qe4 Kg1 Qe3 Rf2 Qg5 h4 Qh6 Qe5 Rxe8 Nf6 Kf7 Nxe8
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 g3 Nc6 Bg2 f5 Nc3 d6 d3 Be7 Rb1 a5 a3 Bd7 b4 Qb8 b5 Nd4 e3 Ne6 Nge2 Qa7 d4 c6 bxc6 Bxc6 d5 Qc5 dxc6 bxc6 Na4 Qa7 Bxc6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 Nf3 O-O Be2 e5 O-O Na6 Re1 c6 Bf1 exd4 Nxd4 Ng4 h3 Qf6 hxg4 Qxd4 Qxd4 Bxd4 g5 Bg4 Be3 Bxe3 Rxe3 Rfd8 f3 Be6 Rd1 Kf8 Be2 Rd7 f4 Nc7 g4 b5 cxb5 cxb5 b3 b4 Na4 Rdd8 f5 Bc8 Rh3 gxf5 exf5 Bb7 Rxh7 Be4 Kf2 Nd5 Bf3 Nc3 Nxc3 Bxf3 f6 Ke8 Re1 Kd7 Rxf7 Kc6 Kxf3 bxc3 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O e4 d6 Be2 c5 O-O Nbd7 d5 Ne8 Bg5 a6 a4 h6 Bf4 Nc7 Qd2 Kh7 Rfe1 Nf6 h3 Nh5 Bh2 Bd7 e5 dxe5 Nxe5 Nf6 Bf3 Rb8 Rad1 Be8 Nd3 b6 a5 Nd7 d6 exd6 Bxd6 Rh8 Nd5 Nxd5 Bxb8 Nxb8 Bxd5 Nc6 Rxe8 Qxe8 axb6 Bd4 b4 cxb4 Nxb4 Nxb4 Qxd4 Nxd5 Qxd5 Qe7 c5 a5 c6 a4 Qd7 Qf6 b7 a3 c7 a2 c8=Q a1=Q Rxa1 Qxa1
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 g3 c5 Nf3 Nf6 Bg2 Nc6 O-O Qc7 c4 cxd4 Nxd4 a6 Nc3 d6 b3 Be7 Bb2 O-O Rc1 Nxd4 Qxd4 Rd8 e4 b6 Nd1 Bb7 Ne3 Bf8 Qd3 Nd7 b4 Rac8 a3 Qb8 f4 Bc6 f5 b5 fxe6 fxe6 Bh3 Re8 Qb3 Bxe4 cxb5 Rxc1 Bxe6 Kh8 Rxc1 Qxb5 Bf7 Re7 Rf1 Qd3 Bc4 Qxb3 Bxb3 Bd3 Nd5 Bxf1 Nxe7 Bxe7 Kxf1 Bd8 Bc4 Bb6 Ke2 h6 a4 Bg1 a5 Ne5 Bxe5 dxe5 Bxa6 Kg8 Bb7 Kf7 Be4 Ke6 h3 Kd7 g4 Bh2 Kd3 Bg3 Kc4 Be1 Kb5 Kc7 Ka4 Kd6 Kb5 Kc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c5 cxd5 Nf6 Nf3 cxd4 Qxd4 Qxd5 Nc3 Qxd4 Nxd4 a6 g3 e5 Nc2 Bd7 Bg2 Bc6 O-O Bxg2 Kxg2 Nc6 Bg5 Be7 Rfd1 Rd8 Bxf6 Rxd1 Rxd1 Bxf6 Ne3 O-O Rd7 b5 Ne4 Rc8 Nxf6 gxf6 Nf5 Kf8 e3 Rd8 Rc7 Nb8 Kf3 Ke8 Ke4 Rd7 Rc8 Rd8 Nd6 Kd7 Rxd8 Kxd8 Kd5 Ke7 b4 Kd7 Ne4 f5 Nc5 Ke7 a3 f6 f4 e4 Nb3 Nd7 Kc6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 d6 d4 f6 c3 Bg4 h3 Bh5 Nbd2 Be7 Qa4 Qd7 d5 O-O-O dxc6 bxc6 Ba6 Kb8 Qb3 Ka8 Qb7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 Bg5 d5 Nbd2 Be7 e3 c5 c3 Nbd7 Bd3 b6 Ne5 Nxe5 dxe5 Nd7 Bxe7 Qxe7 f4 Bb7 Qe2 f6 exf6 gxf6 e4 O-O-O exd5 Bxd5 Be4 Qd6 O-O Qc6 b4 Rhg8 b5 Qb7 c4 Bxe4 Nxe4 Kb8 Rad1 Rg6 g3 Rdg8 Rd3 f5 Nd6 Qc7 Rfd1 Nf6 Qxe6 Ng4 Qxf5 Rf6 Qe4 Qd7 Qc6 Qxc6 bxc6 a6 Re1 Rd8 c7 Kxc7 Ne8 Rxe8 Rxe8 b5 Rd5 Rc6 cxb5 axb5 Rdd8 Kb6 Rb8 Ka5 Rec8
Result: 1-0
Processing game...
FEN: 
Moves: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bg4 h3 Bh5 g4 Bg6 b4 Qb6 Bg2 c6 Bd2 Nbd7 Ne5 Nxe5 dxe5 Rd8 Qe2 Nd5 Na4 Qc7 c4 Nb6 Nb2 e6 f4 Be7 O-O h6 f5 exf5 gxf5 Bh7 f6 Bf8 Bf4 g5 Bh2 Qc8 c5 Na8 Rad1 Nc7 Nc4 Nd5 Rxd5 cxd5 Nd6 Bxd6 exd6 Qe6 Re1 Kd7 Qg4 Qxg4 Re7 Kc8 Rc7 Kb8 hxg4 Bg6 b5 Rc8 Re7 Rhg8 Bxd5 Ka8 d7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nc6 e4 e5 Bb5 d6 d4 Bd7 Nc3 Nf6 O-O Be7 Re1 O-O Bxc6 Bxc6 dxe5 dxe5 Qxd8 Raxd8 Nxe5 Bb4 Nxc6 bxc6 f3 h6 Be3 Rfe8 Rad1 Bxc3 bxc3 a6 Kf2 Nd7 Rd4 Ne5 Ra4 Ra8 Rb1 a5 Rb7 Rec8 Ke2 Kf8 f4 Nd7 e5 Ke7 g4 f6 exf6 gxf6 f5 Kd8 Re4 Ne5 Bxh6 Rcb8 Rxb8 Rxb8 Bg7 Nd7 h4 Rb2 Kd2 Rxa2 h5 Ra1 h6 Rh1 g5 fxg5 f6 Nxf6 Bxf6 Kd7 Bxg5 c5 Rh4 Rxh4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 g6 d4 Nxd4 Nxd4 exd4 Qxd4 Qf6 e5 Qb6 Qxb6 axb6 Nc3 Bb4 Bd2 Ra5 a4 c6 Ne4 Bxd2 Kxd2 Nh6 b4 Ra8 Nd6 Ke7 Bd3 f6 f4 Nf7 Nc4 fxe5 fxe5 Ra6 b5 Ra7 Rhe1 Ke6 Nxb6 Nxe5 c4 d6 Kc3 Kf6 Rad1 Bg4 Rd2 Rd8 Bc2 Bf5 Bxf5 gxf5 Red1 Nf7 Kb4 Ke7 Rf1 Nh6 Re2 Kf6 Rfe1 Ng8 Re8 Rxe8 Rxe8 Kf7 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 g3 c5 Bg2 d5 O-O Nc6 c4 Be7 cxd5 exd5 dxc5 Bxc5 Qc2 Bb6 Nc3 h6 Rd1 O-O Na4 Bg4 Nxb6 axb6 Be3 Re8 Qb3 Ra6 a3 Bf5 Nd4 Bg6 Bf1 Na5 Qb4 Nc4 Bc1 Ne4 b3 Ne5 f3 Nc5 Qc3 Ra8 Bb2 Rc8 Rac1 Qd7 Qd2 Qe7 Rc3 Nc6 Rdc1 Rcd8 Ba1 Nxd4 Qxd4 Ne6 Qxb6 d4 Rc4 Qxa3 Qb4 Qa7 Qa4 Qb6 Qb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d6 Nf3 O-O O-O Nbd7 Qc2 c6 Rd1 Re8 Nc3 e5 e4 a5 d5 Qc7 Ne1 Nc5 Be3 Nfd7 b3 Na6 a3 Ndc5 Rab1 a4 b4 Nb3 Nxa4 Nd4 Bxd4 exd4 Nb2 c5 Nbd3 f5 Nf4 fxe4 Bxe4 Rxe4 Qxe4 Bf5 Qe2 Bxb1 Rxb1 cxb4 axb4 Rc8 Ned3 Qf7 Ne6 Nc7 Nxc7 Rxc7 Rc1 b5 cxb5 Rxc1 Nxc1 Qxd5 Qe8 Bf8 Qc6 Qe5 Nd3 Qf5 Qc4 d5 Qxd4 Bg7 Qe3 d4 Qe8 Bf8 b6 Qd5 b5 Qb7 Qc6 Qb8 b7 Kf7 Qc8
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 f5 d3 Nf6 e4 fxe4 dxe4 e6 e5 Nd5 Bd3 Be7 c4 Nb6 Nc3 d5 exd6 cxd6 O-O N8d7 b3 Nf6 Qe2 O-O Bb2 e5 c5 dxc5 Nxe5 Kh8 Rad1 Qe8 Rfe1 Bd6 Nb5 Bb8 Bb1 Bd7 Nd6 Bxd6 Rxd6 Rc8 Qd2 Qh5 Nxd7 Nbxd7 Rxd7 Nxd7 Qxd7 Qf7 Qxf7 Rxf7 h4 b5 h5 Kg8 g4 Rfc7 Bf5 Rd8 Re3 h6 Re6 c4 bxc4 bxc4 Be5 Rb7 Rc6
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 Nf3 Nc6 e3 Be7 a3 a5 Qc2 O-O b3 d5 cxd5 Nxd5 Bb2 Bf6 Ne4 Bf5 d3 Bg6 Be2 Re8 h4 h5 Nfd2 Be7 Nf3 Bf8 Rd1 Kh8 Nfd2 Qe7 Ng3 Nf6 Nde4 Ng4 Bxg4 hxg4 h5 Bh7 h6 gxh6 Nh5 f5 Nef6 Red8 Nxh7 Qxh7 d4 exd4 exd4 Re8 Kf1 Bg7 d5 Ne5 Qxc7 Rac8 Qxg7 Qxg7 Nxg7 Kxg7 Re1 Kf6 f4 gxf3 gxf3 Rc2 Ba1 f4 Rh5 Ra2 Rhxe5 Rxe5 Bxe5 Kf5 Bc7 Rd2 d6 h5 Re7 Kf6 Rh7 Ke6 Rxh5 Rd3 Ke2 Rxb3 Rxa5 Re3 Kf2 Rb3 Re5 Kd7 a4 Rb2 Ke1 Rb3 Ke2 Rb2 Kd3 Rb4 a5 Ra4 Rb5 Kc8 Rf5 Kd7 Rf7 Kc8 Bb6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 f5 d3 d6 g3 Nf6 Bg2 c6 O-O g6 e4 fxe4 dxe4 e5 Nbd2 Be7 Ne1 Be6 Nd3 Nbd7 b3 Qc7 Bb2 O-O Kh1 Rae8 f4 Rf7 c4 Bg4 Qe1 exf4 gxf4 Bf8 Qf2 Rfe7 Rae1 Bg7 h3 Be6 Nf3 Nh5 Bxg7 Rxg7 Ng5 a5 Bf3 Nhf6 e5 dxe5 fxe5 Nh5 Bxh5 gxh5 Nxe6 Rxe6 Nf4 Ree7 e6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e6 Bxc4 Nc6 Nf3 Nf6 Nc3 Bb4 e5 Nd5 Bd2 Nxc3 bxc3 Ba5 O-O O-O Bd3 g5 Bxg5 Ne7 Qc1 Qd7 Bf6 Nf5 Bxf5 exf5 Qg5
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 d6 d4 g6 e4 Bg7 Nf3 O-O Be2 Nbd7 O-O e5 Qc2 c6 Rd1 Qc7 d5 cxd5 cxd5 Nc5 Nb5 Qe7 Nd2 a6 Nc3 b5 b4 Nb7 a4 bxa4 Rxa4 a5 Nc4 Qc7 Nb5 Qd8 Be3 Ra6 Rda1 Bd7 Nxa5 Nxa5 Rxa5 Rxa5 bxa5 Bxb5 Bxb5 Qc8 Qxc8 Rxc8 a6
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 g3 Nf6 Bg2 g6 Nh3 Bg7 Nf4 d6 d5 c6 c4 Bd7 Nc3 Na6 h4 Rg8 h5 g5 Nd3 h6 e4 Qc7 Bd2 e5 b4 cxd5 cxd5 f4 Rc1 Rc8 Bf1 Qb6 gxf4 exf4 f3 Ke7 a3 Rge8 Qe2 Kf8 Qf2 Qxf2 Kxf2 Rc7 Bh3 Rec8 Bxd7 Rxd7 Nb5 Rxc1 Rxc1 Nxh5 Bc3 Nf6 Bd4 b6 a4 Ke7 Rc6 Nb8 Rc8 Rb7 e5 dxe5 Nxe5 Nxd5 Rxb8 Rxb8 Nc6 Kf7 Nxb8 Bxd4 Nxd4 Nxb4 Nbc6 Nd3 Ke2 Nc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 c6 e4 d5 e5 Ng4 Nf3 Bg7 cxd5 cxd5 Qb3 e6 h3 Nh6 g4 Nc6 Bb5 a6 Bxc6 bxc6 Bg5 Qc7 Na4 Ng8 Rc1 Bd7 Qa3 f6 Bf4 fxe5 Bxe5 Bxe5 Nxe5 Nf6 Nc5 Qc8 Ncxd7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 Bf5 cxd5 cxd5 Qb3 Qd7 Ne5 Qd8 Qxb7 Nbd7 e3 a6 Nxd7 Bxd7 Nxd5 Nxd5 Qxd5 e6 Qb3 Rb8 Qd1 Qa5 Bd2 Bb4 b3 O-O Bc4 Bc6 Bxb4 Qxb4 Qd2 Qxd2 Kxd2 Bxg2 Rhg1 Bb7 Rg5 Rfd8 Ra5 Ra8 Rc1 Rdc8 Bd3 Rxc1 Kxc1 g6 Kd2 f5 b4 Kf7 Rc5 Rc8 Kc3 g5 Rxc8 Bxc8 Kc4 Bd7 Kc5 Kg6 Bxa6 Kh5 b5 Kh4 b6 Kh3 b7 e5 b8=Q Kxh2 Qxe5 f4 exf4 Kh3 fxg5 Ba4 Qg3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 Nbd7 Bd3 O-O Nge2 Re8 Qc2 c6 O-O Nf8 f3 g6 Rad1 Ne6 Bh4 Ng7 Bf2 Be6 Kh1 Bd6 e4 dxe4 Nxe4 Nd5 Nxd6 Qxd6 Qd2 Bf5 Bc4 Nb6 Bb3 Be6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 f5 Nc3 e6 e4 fxe4 Nxe4 Be7 Nf3 Nf6 Bd3 d6 Qe2 O-O h4 Qe8 Nfg5 Nxe4 Qxe4 Rf5 g4 d5 Qe2 Rf6 Bxh7 Kh8 Bd3 Qf8 c3 Qg8 Nf3 Nc6 g5 Rf5 Ne5 Nxe5 Qh5
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6 d4 Be7 Nf3 h6 Bd3 Nd7 O-O O-O Qe2 Re8 Nd2 g6 f4 Qg7 Nf3 b6 Ne5 Nxe5 fxe5 Bb7 Be3 Rf8 Qd2 Kh7 Rf2 f5 exf6 Rxf6 Rxf6 Bxf6 Rf1 Rf8 Rf4 Bg5 Rxf8 Bxe3 Qxe3 Qxf8 Qxe6 Qg7 h4 h5 Kh2 Qh6 Qf7 Kh8 Bxg6 a5 Qf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 O-O Ne2 Nc6 O-O d6 d5 Ne7 e4 Bxc3 Nxc3 e5 a3 Ng6 g3 Bh3 Re1 h5 f3 Nh7 b4 b6 Ra2 Qe7 Rf2 Ng5 Qc2 Bd7 Bxg5 Qxg5 Qc1 f6 Kf1 Rae8 Qe3 Rf7 f4 Bh3 Ke2 exf4 gxf4 Qg4 Kd2 Rfe7 Rf3 Qc8 Rxh3 Nxf4 Rg3 h4 Rgg1 Nh3 Rgf1 cxb4 axb4 Ng5 Ra1 Qg4 Rf2 f5 Be2 Nxe4 Nxe4 Qxe4 Qxe4 fxe4 Ke3 Kh7 Rg1 a5 bxa5 bxa5 Rf4 Rb8 Rxh4 Kg8 Rxe4 Rb3 Bd3 Rxe4 Kxe4 a4 Ra1 a3 c5 dxc5 d6 Kf8 Bc4 Rb4 Kd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 d4 Nf6 O-O cxd4 Qxd4 a6 Bxd7 Bxd7 Bg5 h6 Bxf6 gxf6 c4 e6 Nc3 Rc8 Kh1 h5 a4 h4 h3 Be7 b4 a5 b5 Qc7 Nd2 Qc5 Qd3 Rg8 Rae1 Qg5 Rg1 Qf4 Ref1 b6 Ne2 Qh6 c5 Rxc5 Nc4 Kf8 Nxb6 Be8 f4 f5 exf5 Rxf5 Rc1 Kg7 g4 Rc5 Rxc5 dxc5 Nc8 Bf8 Qd8 Qg6 f5 Qh6 g5 Qh5 Rg4 exf5 Nf4 Qh8 Qf6 Kh7 Rxh4
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 a6 b3 Bb4 Bd2 Nbd7 Bd3 O-O O-O Bd6 e4 dxc4 bxc4 e5 c5 Bc7 Na4 exd4 h3 Re8 Re1 h6 Rb1 Nh7 Bc4 Qf6 Rb3 Nhf8 Bc1 Ng6 Nxd4 Nde5 Bf1 Rd8 Bb2 Nh4 Ree3 Qg5 g3 Nhg6 Qc2 h5 Bg2 h4 f4 Nxf4 gxf4 Qxf4 Nb6 Ng4 hxg4 Bxg4 Nxa8 Qh2 Kf1 Bg3 Rxg3 hxg3 Qd3 Bh3 Qxg3 Qh1 Kf2 Bxg2 Qxg2 Qh4 Ke2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 d5 e3 c5 c3 Nbd7 Qb3 O-O Be2 b6 O-O Bb7 a4 a6 Rfd1 Qb8 Bf4 Qa7 Ne5 e6 Qa3 Rac8 b4 Nxe5 Bxe5 c4 Re1 Bc6 Bf3 Qb7 Qb2 b5 axb5 axb5 Bxf6 Bxf6 e4 Ra8 e5 Bg5 g3 Rxa1 Rxa1 f5 Be2 Ra8 f4 Be7 Nf3 Rxa1 Qxa1 Qa8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 b6 Nc3 Bb4 Qb3 c5 a3 Bxc3 Qxc3 O-O g3 cxd4 Qxd4 Nc6 Qh4 Bb7 Bg2 Rc8 Bd2 Na5 Bxa5 bxa5 b4 Qc7 c5 d6 Rc1 axb4 axb4 dxc5 Rxc5 Qb6 Rxc8 Rxc8 O-O Ba6 Ne5 Bxe2 Re1 Bb5 Qf4 h6 h4 Be8 g4 Rd8 g5 Rd4 Qf3 hxg5 hxg5 Nd5 Rc1 Qb8 Qg3 Nf4 Qe3 Qd8 Nf3 Rxb4 Bf1 Nd5 Qxa7 Rg4 Kh1 Nf4 Qb7 Kh7 Qe4 f5 gxf6 Bg6 Qb7 Qxf6 Re1 Nd5 Bg2 Nf4 Bf1 Nd3
Result: 0-1
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 Nc3 Nd4 Ng1 e6 e3 Nf5 Nf3 Nf6 d4 cxd4 exd4 Bb4 Bd3 O-O O-O b6 Ne2 Bb7 d5 Ne7 Bg5 Ng6 a3 Bd6 Nc3 Re8 Re1 a6 Ne4 Be7 d6 Bf8 h4 h6 h5 hxg5 hxg6 fxg6 Ne5 Rc8 b4 a5 Rc1 axb4 axb4 g4 c5 bxc5 bxc5 Nxe4 Bxe4 Bxe4 Rxe4 Qg5 Rcc4 g3 fxg3 Red8 g4 Ra8 Qc1 Qxc1 Rxc1 g5 Rb4 Rdc8 Rbc4 Ra7 Rf1 Ra5 Rfc1 Rd8 Re4 Ra2 Re3 Rd2 Rf1 g6 Nxg6 Kg7 Nxf8 Rxf8 Rxf8 Kxf8 Re5 Kf7 Rxg5 Rc2 Rh5 Kf6 Rh7 Rxc5 Rxd7 Rg5 Rd8 Ke5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Qc2 e5 Rd1 Qe7 Nc3 c6 e4 exd4 Nxd4 Re8 b3 Nc5 f3 Nfd7 Rb1 Ne5 Nce2 a5 a3 h5 h4 Bd7 Be3 a4 b4 Ne6 c5 d5 exd5 cxd5 Bf2 Nc4 Qc3 Rac8 Rbc1 b6 cxb6 Nxb6 Qd3 Nc4 Re1 Qd6 f4 Nb2 Qd2 Nc4 Qd3 Nb2 Qd2 Nc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 c6 Nc3 g6 e4 d5 e5 Ne4 Bd3 Nxc3 bxc3 Bg7 Ne2 O-O cxd5 cxd5 h4 Be6 h5 Qc8 Bd2 Bf5 Qb1 Nc6 Bxf5 Qxf5 Qxf5 gxf5 h6 Bxe5 dxe5 Nxe5 Nd4 e6 Ke2 Nc4 Rae1 Kh8 Bg5 f6 Nxe6 fxg5 Nxf8 Rxf8 Kd3 Nd6 f3 Kg8 Re7 Rf7 Rhe1 Kf8 R7e6 Rd7 Rf6 Kg8 Ree6 Nf7 Re8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 c6 e3 Nbd7 Qc2 Bd6 Be2 O-O O-O e5 cxd5 cxd5 dxe5 Nxe5 Rd1 Nxf3 Bxf3 Qc7 h3 Bh2 Kh1 Be5 Bd2 Be6 Qd3 Qd7 Be1 Rfd8 Kg1 Rac8 Rac1 d4 exd4 Bxd4 Qe2 h6 b3 a6 Rc2 b5 Rcd2 Qa7 Ne4 Nxe4 Qxe4 Bb6 Rxd8 Rxd8 Rxd8 Bxd8 Bc3 Qc5 Qe5 Qxe5 Bxe5 a5 Be4 f6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 Nf6 e4 c5 e5 Ng8 Nf3 Nc6 d4 cxd4 Nxd4 Nxe5 Ndb5 a6 Nd6 Bxd6 Qxd6 f6 Be3 Ne7 Bb6 Nf5 Qc5 d6 Qa5 Qd7 f4 Nc6 Qa3 e5 Bd3 O-O O-O exf4 Rxf4 Nfe7 Rd1 Ng6 Rff1 Nge5 Be4 Qf7 b3 Be6 Qxd6 Kh8 Qc7 Qxc7 Bxc7 Rf7 Bb6 Re8 h3 Rd7 Nd5 Rc8 g4 Ng6 Kh2 Nce5 a4 Rd6 a5 Nd7 Nc7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 c4 Nf6 Nc3 Qc7 Bd3 Be7 f4 d6 Qe2 Nc6 Nf3 Nd7 a3 Bf6 Be3 Bxc3 bxc3 e5 f5 Ncb8 O-O Nc5 Bc2 Nbd7 Rfd1 Nf6 Nd2 Bd7 Bg5 Bc6 Qf3 O-O-O Re1 h6 Bh4 Rdg8 Kh1 Nfd7 Nf1 g5 Bf2 h5 Qd1 h4 Qb1 Nf6 Bxc5 dxc5 Ne3 Qa5 Qb2 h3 g3 Nxe4 Bxe4 Bxe4 Kg1 Rd8 Ng4 Bxf5 Nxe5 Qc7 Qf2 Be6 Rab1 Rd6 Rb2 Rhd8 Rbe2 f6 Ng6 Bxc4
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Bd6 O-O O-O c4 Bxe5 dxe5 Nc6 cxd5 Qxd5 Qc2 Nb4 Bxe4 Nxc2 Bxd5 Bf5 g4 Bxg4 Be4 Nxa1 Bf4 f5 Bd5 Kh8 Rc1 c6 Bg2 Rfd8 Nd2 Rxd2 Bxd2 Rd8 Bc3 Rd1 Rxd1 Bxd1 f4 Nc2 Kf2 Kg8 a4 a5 Bxa5 Nd4 Bf1 Bb3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 d6 f4 Nc6 Nf3 g6 Bb5 Bd7 O-O Bg7 d3 a6 Bxc6 Bxc6 Qe1 Qd7 a4 b6 b3 Nf6 h3 Nh5 Bd2 f5 exf5 gxf5 Qh4 Nf6 Rae1 O-O-O a5 b5 b4 cxb4 Na2 Nd5 Nxb4 Bf6 Qf2 Nxb4 Bxb4 Rhg8 Re2 Rg6 Qe1 Rdg8 Kh2 e5 fxe5 Bxe5 Kh1 Qb7 d4 Rxg2 Rxg2 Rxg2 Kxg2 Qg7 Kh1 Bf4 Qe6 Bd7 Qd5 Qg3 Qa8 Kc7 Re1 Qxh3 Kg1 Qg3 Kf1 Qh3 Ke2 Bc6 Qa7 Kc8 Rf1 Qg2 Ke1 Bg3 Kd1 Qxf1 Ne1 Bxe1
Result: 0-1
Processing game...
FEN: 
Moves: c4 f5 g3 Nf6 Nc3 g6 Bg2 Bg7 d3 d6 e3 O-O Nge2 e5 O-O Nbd7 Rb1 Kh8 b4 a6 Bb2 Nh5 f4 Ndf6 Qd2 c6 a4 Qe7 b5 axb5 axb5 Bd7 h3 Rfe8 Ra1 h6 e4 Rxa1 Rxa1 Qd8 exf5 Bxf5 Kh2 exf4 Nxf4 Nxf4 Qxf4 Nh5 Qd2 Kh7 Ne4 d5 Bxg7 Kxg7 Qb2 Kg8 cxd5 cxd5 Nc3 Qc7 Bxd5 Kh7 Rg1 Re3 Ne4 Bxe4 Bxe4 Nxg3 Bxg6 Kxg6 Qf2 Rxd3 Rxg3 Rxg3 Qxg3 Qxg3 Kxg3 Kg5 b6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 e6 Nc3 d5 d4 Be7 Bg5 h6 Bxf6 Bxf6 e3 O-O Rc1 c6 Bd3 Nd7 cxd5 exd5 b4 a6 a4 Be7 b5 axb5 axb5 Nf6 bxc6 bxc6 O-O c5 dxc5 Bxc5 Nb5 Bb6 Qb3 Bd7 Nbd4 Rb8 Qa3 Ra8 Qb2 Rb8 Qa3 Ra8 Qb4 Rb8 Qd6 Bxd4 Nxd4 Qb6 Qf4 Rbc8 h3 Qb8 Qxb8 Rxb8 Rc7 Rfc8 Ra7 Ra8 Rfa1 Rc1 Rxc1 Rxa7 g4 Kf8 Rb1 Ra3 Bf5 g6 Bxd7 Nxd7 Rb5 Nf6 Rb8 Kg7 Rb7 Ne4 Ne6 Kg8 Nd8 Nd6 Rb6 Ne4 Rb7 Nd6 Rd7 Ra6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 f3 Ng8 Qc2 axb4 axb4 Ndf6 c5 Nh5 Nc4 Nf4 cxd6 cxd6 Nb5 Ra6 Rb3 fxe4 fxe4 Bd7 Nc3 Ra1 Bxf4 Rxf1 Bxf1 exf4 Qf2 Nh6 h3 Nf7 Ne2 Ng5 Nxf4 Qf6 g3 Nxe4 Qe3 Nxg3 Qxg3 Qxf4 Qxf4 Rxf4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 f3 Ng8 Qc2 Ngf6 Nb5 axb4 axb4 Nh5 g3 Ndf6 c5 Bd7 Rb3 fxe4 fxe4 Bh3 Re1 Bh6 Nf3 Bxc1 Rxc1 Ng4 Qd2 Qe7 Rbc3 h6 Bf1 Bxf1 Rxf1 g5 cxd6 cxd6 Nc7 Ngf6 Nxg5 hxg5 Nxa8 Nxe4 Rxf8 Qxf8 Qe2 Nhf6 Ra3 Qc8 Kg2 Qc1 Rf3 b5 Nb6 Qc7 Qxb5 Qc2 Kg1 Qd1 Rf1 Qd4 Kh1 Nf2 Kg2 N2g4 Qe2 Qxb6 Re1 Qxb4 h3 Nh6 Rf1 Kg7 Kh2 Qe4 Qa2 Qxd5 Qa7 Kg6 Qa1 Nf5 Qb1 e4 Qb2 Qe5 Qxe5 dxe5 Re1 Nd4 Kg2 Kf5 Rf1 Ke6 Ra1 e3 Ra6 Kf5 g4 Kg6 Ra3 Nd5 Ra6 Kf7 Ra5 Ke6 Ra6 Kd7 Ra5 Kc6 Ra6 Kc5 Kf1 Nf4 Ra5 Kd6 Ra6 Kd5 Ra3 Ke4 Ra2 Kf3 h4 gxh4 g5 e2 Rxe2 Ndxe2 Ke1 h3 Kd2 h2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rfe1 Bd7 a4 b4 Nce2 Kh8 Bg5 Bxg5 Qxg5 Nf6 Ng3 Nc6 Nxc6 Bxc6 Rad1 Rad8 Rd3 h6 Qf4 Qe7 Qh4 Rd7 Nh5 Nxh5 Qxh5 Qd8 h3 Qb6 a5 Qb7 Qh4 Qb5 Red1 Rfd8 Qf4 Qxa5 Rxd6 Qg5 Rxd7 Qxf4 Rxd8 Kh7 R8d4 a5 c3 a4 Bc2 bxc3 e5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Nc6 Bb5 Bd6 c4 dxc4 d5 a6 Ba4 b5 dxc6 bxa4 O-O Ne7 Qxa4 O-O Qxc4 Be6 Qc2 Bf5 Qa4 Bd3 Re1 Bb5 Qc2 Nxc6 a3 Qd7 Nc3 Bd3 Qa4 Rfe8 Be3 Ne5 Qxd7 Nxd7 Rad1 Bg6 Nd5 Ne5 Nxe5 Rxe5 f3 Bc2 Rc1 Bb3 Nxc7 Rc8 Rc3 Rxc7 Rxb3 f5 Kf1 Rd5 Rb8 Kf7 Rd8 Re7 Bf2 Rxe1 Bxe1 Ke6 Ra8 Bxh2 Rxa6 Rd6 Rxd6 Bxd6 Ke2 Kd5 Kd3 h5 a4 g5 b4 g4 a5 g3 a6 Kc6 b5 Kb6 Ba5 Ka7 Bd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 Nf3 O-O e3 c5 Be2 cxd4 exd4 d5 O-O Nc6 h3 Bf5 cxd5 Nxd5 Qb3 Be6 Qxb7 Nxd4 Nxd4 Bxd4 Bh6 Rb8 Qa6 Rxb2 Nxd5 Qxd5 Bxf8 Kxf8 a4 Rb3 Rad1 Rg3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 c6 Bd3 e5 d5 cxd5 cxd5 Nh5 Nge2 f5 exf5 gxf5 O-O Nd7 Rc1 Nc5 Bc4 a6 b4 Nd7 a4 Qe8 Kh1 Ndf6 b5 Qg6 Bd3 e4 Bc2 Bd7 bxa6 bxa6 Rb1 Rae8 Qd2 Kh8 Rb6 Rg8 Rg1 Qf7 Rxa6 f4 Bd4 Ng3 Nxg3 fxg3 Qf4 gxh2 Rf1 exf3 Qxf3 Ref8 Rxd6 Bg4 Rxf6 Bxf3 Rxf7 Bxg2 Kxh2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Bd6 O-O O-O c4 Bxe5 dxe5 Nc6 cxd5 Qxd5 Qc2 Nb4 Bxe4 Nxc2 Bxd5 Bf5 g4 Bxg4 Be4 Nxa1 Bf4 f6 Nc3 fxe5 Be3 Bf3 Rxa1 Bxe4 Nxe4 b6 b4 a5 b5 Rad8 Rc1 Rf7 a4 h6 Kg2 Rd3 Rc4 Kf8 Nd2 Ke8 Nf3 Re7 Re4 Rb3 Nxe5 g5 Kf3 Re6 Ke2 Rb4 Rxb4 axb4 Bd4 b3 Kd3 Rf6 Kc3 Rf4 Nc6 Rf3 Be3 Rh3 Kxb3 Rxh2 Kc4 Rh4 Nd4 Kd7 Kd5 Rh1 Nf5 Ra1 Nxh6 Rxa4 Nf5 g4 Nd4 Ra8 Ke5 Rf8 Bf4 Rf7 Bg3 Rf8 Bh4 Rf7 Ne6 Rf3 Bg3 Rf7 Ng5 Re7 Kd5 Re1 Ne4 Rb1 Nc3 Rb4 Be5 Rb2 Bd4 Rb4 Be3 Kc8 Ke5 Kb7 Kf5 Rb3 Ne2 Rxb5 Kxg4 Re5 Nc3 c5 Bd2 Kc6 f4 Re8 f5 b5 Kf4 b4 Ne4 Rb8 Bc1 c4 f6 Kd5 Kf5 c3 Bh6 Rb7 Bf8 c2 Bh6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5 Qc2 c6 Rd1 Qe7 b3 exd4 Nxd4 Re8 Bb2 Nc5 e3 a5 a3 h5 b4 Nce4 b5 Bd7 Rac1 h4 a4 hxg3 hxg3 Nxf2 Qxf2 Ng4 Qf3 Nxe3 Re1 Bxd4 Nd5 Ng4 Bxd4 Qxe1 Rxe1 Rxe1 Bf1 cxd5 Qxd5 Rae8 Bf2 Be6 Qxb7 Rc1 Qc6 Rc8 Qe4 R8xc4 Qa8 Kh7 b6 Rb4 Qxa5 Rbb1 Kg2 Rc2
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 c5 O-O c4 Re1 Be7 Bf1 O-O Bg5 Bg4 h3 Bxf3 Qxf3 Nc6 c3 Qd7 Nd2 Rae8 b3 b5 bxc4 bxc4 Rab1 Bd8 h4 Rxe1 Rxe1 Re8 Rb1 h6 Bxf6 Bxf6 g3 Ne7 Qd1 g5 hxg5 hxg5 Bg2 g4 Nf1 Bg5 Rb2 Kg7 Qb1 Rc8 Re2 Rc6 Re5 f6 Re2 Rb6 Qd1 Nf5 Re1 Nh6 Qe2 Kf7 f4 gxf3 Qxf3 Rd6 Re5 Kg7 Rxd5 Rxd5 Qxd5 Qa4 Qb7 Kg6 Bc6 Qa5 Be8 Kf5 Qh7 Kg4 Qe4 Kh3 Bd7 f5 Qg2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 a6 a4 Nc6 Be3 Be7 O-O O-O f4 Qc7 Kh1 Re8 Bf3 Rb8 Qd2 Bd7 Rad1 Na5 b3 Rec8 Nde2 Be8 Ba7 Ra8 Bd4 Nc6 Bxf6 Bxf6 Qxd6 Qb6 e5 Be7 Qd2 Bb4 Qd3 Rd8 Qe4 Qc5 Na2 Bd2 c3 a5 Nac1 Bxc1 Rxc1 Rac8 Ng3 Nb4 Rcd1 Qxc3 Qxb7 Qxb3 Rxd8 Rxd8 Qc7 Rd4 Ne4 Qd3 Rg1 Rxe4 Qc8 Kf8 Qc5 Kg8 Qe7 g6 Qxe8 Kg7 Bxe4 Qxe4 Rf1 Nd3 Qb5 Qe3 h3 Nf2 Kh2 Qxf4 Kg1 Nxh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 d5 a5 Be3 Ng4 Bg5 f6 Bh4 Na6 Nd2 h5 a3 Bd7 h3 Nh6 Rb1 Nc5 b4 axb4 axb4 Na4 Qc2 Nxc3 Qxc3 g5 Bg3 h4 Bh2 f5 c5 g4 c6 Bc8 hxg4 fxg4 cxb7 Bxb7 O-O Qg5 Ra1 Rxa1 Rxa1 Bxd5 Ra7 Be6 Rxc7 Qf6 Qe3 Nf7 Bc4 Bh6 Qe2 Bxd2 Bxe6 Qxe6 Qxd2 Qb3 Kh1 Qb1 Bg1 Qxe4 Qc2 Qxc2 Rxc2 d5 b5 d4 f3 g3 b6 Rd8 Rc6 Nd6 Rc7 Rb8 Rc6 Nf5 Re6 Ne3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Qc7 Nc3 e6 Be3 a6 Bd3 Nf6 O-O Ne5 h3 Bc5 Kh1 d6 f4 Nc6 e5 Nxe5 fxe5 dxe5 Bb5 axb5 Ndxb5 Qb6 Bxc5 Qxc5 Nd6 Ke7 Rxf6 gxf6 Nce4 Qd4 Qh5 Rf8 Rd1 Qe3 Qh4 Qf4 Qe1 Ra4 Qc3 Rd4 Rxd4 Qf1 Kh2 exd4 Qc5 Kd7 Nb5 Qf4 g3
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Bg4 h3 Bh5 Qe2 Qe7 Be3 Nc6 Nc3 O-O-O g4 Bg6 O-O-O f6 a3 Qd7 Nd2 f5 Nb3 Nf6 f3 Bd6 Qd2 Rhe8 Bg5 fxg4 hxg4 Qf7 Nb5 Kb8 Nxd6 cxd6 Bd3 Bxd3 Qxd3 h6 Bd2 Re6 Na5 Nxa5 Bxa5 Rde8 Bd2 Nd7 Rde1 Nf8 Rxe6 Rxe6 Rh5 Rf6 f4 Ne6 f5 Nd8 b4 Nc6 b5 Ne7 a4 Nc8 a5 Qe8 Rh3 Rf7 Re3 Re7 Rxe7 Qxe7 Qf3 Qf7 Bb4 Kc7 Qc3 Kd8 Qf3 Kc7 Kd2 Kd8 Kd1 Kc7 Be1 Ne7 a6 b6 Bh4 g5 Bf2 Qf6 Qh1 Kd7 c4 Ke8 Kd2 Kf7 cxd5 Kg7 Kd3 Qf7 Bg3 Qe8 Kc4 Nxf5 gxf5 Qe2 Kc3 Qe3 Kc4 Qe2 Kc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Bd3 Nbd7 O-O Qb6 Be3 Qxb2 Ndb5 axb5 Nxb5 Ra5 Rb1 Rxb5 Rxb2 Rxb2 Qa1 Rb6 Bxb6 Nxb6 Qc3 Be7 Rb1 Nfd7 Qxg7 Bf6 Qh6 Ke7 Bb5 Rg8 Rd1 e5 f5 Nc5 Rxd6 Bg5 Qxh7 Nxe4 Rxb6 Rd8 Bd3 Be3 Kf1 Bxb6 Bxe4 Rd4 c3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 Rb1 Nd7 a3 f5 b4 Kh8 f3 Ng8 Qc2 Ngf6 Nb5 axb4 axb4 Nh5 g3 Ndf6 c5 Bd7 Rb3 Nxg3 hxg3 Nh5 f4 exf4 c6 bxc6 dxc6 Nxg3 Rxg3 fxg3 cxd7 g2 Rf3 Qxd7 Bb2 fxe4 Rxf8 Rxf8 Bxg7 Qxg7 Qxe4 Qf6 Nf3 Qf4 Qe7 Rf7 Qe6 Rf6 Qe8 Rf8 Qe7 Rf7 Qe6 Rf6 Qb3 g5 Nxc7 g4 Nd5 Qc1 Qd1 Qxd1 Bxd1 Rf5 Ne3 Rf4 Ne1 Rxb4 Bxg4 h5 Bf3 d5 N3xg2 h4 Nd3 Ra4 Ngf4 Kg7 Kg2 Kf6 Bxd5 Ra5 Bc6 Ra6 Bb7 Ra3 Be4 Ra4 Bd5 Ra5 Bc6 Ra6 Bf3 Kg5 Bb7 Ra1 Bc8 Ra4 Kf3 Rc4 Bd7 Kf6 Kg4 Rd4 Bc6 Rd8 Kxh4 Rg8 Be4 Rg1 Nh5 Ke6 Ng3 Kf6 Kg4 Ra1 Bd5 Ra5 Bf3 Ra1 Kf4 Ke6 Nc5 Kd6 Nge4 Ke7 Ke5 Rf1 Bg4 Rg1 Be6 Re1 Bc8 Rc1 Kd4 Rd1 Nd3 Kf7 Ke3 Ra1 Kf4 Ke7 Nb4 Rc1 Nd5 Kf7 Bd7 Rf1 Ke5 Ra1 Ng5 Kg6 Nf3 Kg7 Bg4 Kg6 Nf4 Kg7 Nd4 Re1 Kf5 Rc1 Be2 Re1 Bh5 Ra1 Nfe6 Kh6 Be8 Ra8 Bc6 Ra1 Kf6 Kh7 Ng5 Kh8 Nde6 Ra6 Be8 Ra8 Bh5 Ra1 Bg6 Rf1 Ke7 Ra1 Nf7 Kg8 Nh6 Kh8 Nf5 Ra7 Kf6 Ra1 Ne3 Re1 Nd5 Rg1 Bf5 Rf1 Ndf4 Ra1 Ng6 Kg8 Ne7 Kh8 Ng5 Ra6 Kf7 Rf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 g6 Nf3 Bg7 Be2 O-O O-O e6 b4 Nbd7 Bb2 dxc4 Bxc4 b6 a4 Bb7 Ba3 Rc8 Rb1 Re8 Qe2 Nd5 Nxd5 cxd5 Bb5 Re7 a5 Rc7 Rfc1 Nb8 a6 Bc8 Bd3 Rxc1 Rxc1 Rc7 b5 Rxc1 Bxc1 Qc7 Bd2 f5 Qe1 Nd7 Bb4 Bf8 h4 Nf6 Ne5 Nd7 Nf3 Nf6 Ne5 Nd7 Nc6 Bxa6 Qc1
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5 c6 Nxe5 Re8 O-O-O Na6 Rd6 Be6 f4 h6 Bxf6 Bxf6 Rhd1 Bxe5 fxe5 Rac8 Kd2 Re7 Ke3 Rce8 Rd8 Kf8 h4 Nc5 Rxe8 Rxe8 b4 Nd7 Kf4 Ke7 Nd5 cxd5 cxd5 Rc8 dxe6 fxe6 Rd2 Rf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 O-O O-O Bb6 Kh1 Rd8 Qh5 h6 Nd2 d5 exd5 Nxd4 cxd4 Bf5 Qf3 Qg6 Bf4 Qg4 Qxg4 Bxg4 f3 Bf5 g4 Bh7 d6 cxd6 Rae1 Kf8 d5 Ba5 Rd1 Rac8 b3 a6 a4 Bb4 Ne4 Bxe4 fxe4 Ng6 Bg3 Re8 Rf2 Rcd8 Bd3 Bc3 Rc2 Be5 Bf2 Bf4 Rc7 Re7 Rc2 Ne5 Be2 Nd7 Bf3 Nf6 Rd4 Rde8 Kg2 g5 Rb4 Kg7 Bd4 Kg6 Bxf6 Kxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 c6 Nc3 d5 d4 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qe2 Bg6 e4 Bxc3 bxc3 Nxe4 Ba3 Qc7 Nd2 Ndf6 Nxe4 Nxe4 Rfe1 O-O-O Qb2 Rhe8 f3 Nd6 Bf1 Kb8 a5 Nc8 Bc5 f6 Ra4 e5 Rea1 Rd7 Rb4 Ka8 Bb6 Qb8 dxe5 Rxe5 Bf2 Qd6 Rd4 Rd5 Rxd5 Qxd5 Qb4 Qd6 Qa4 Rd8 Re1 Ne7 a6 b6 Rxe7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 e6 Nf3 a6 g3 b5 Bg2 Bb7 d3 d6 O-O Nd7 a3 Rc8 Bd2 Ngf6 Nh4 Be7 e5 Nd5 exd6 Bxh4 Nxd5 Bxd5 Bxd5 Bf6 Qe1 Qb6 Rb1 O-O Bg2 Qxd6 Qe2 Rfd8 Rfe1 Nb6 Bf4 Qe7 h4 a5 c3 h6 Rbc1 b4 axb4 axb4 Be4 bxc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g6 f4 f6 Ba3 Qf7 Qd2 Nb6 c5 Bxf1 cxb6 axb6 e6 dxe6 Bxf8 Rd8 Qb2 Bxg2 Qxg2 Kxf8 Qxc6 Rd6 Qc3 Kg7 Nd2 Rhd8 O-O-O Qe8 Qxc7 R8d7 Qc2 Qb8 Nc4 Rd5 Qf2 Qc7 Qxb6 Qxf4 Qe3 Qg4 Rdg1 Qh4 Rg3 e5 Rh3 Qg4 Rg1 Rd1 Rxd1 Qxd1 Kb2 h5 Rg3 Qh1 Qf2 h4 Qg2 Qxg2 Rxg2 g5 a4 Kg6 a5 e4 b4 h3 Rg3 Rh7 a6 f5 Ra3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Na6 Be3 c6 dxe5 dxe5 h3 Qe7 Qc2 Nh5 Rfe1 Nc7 Rad1 Ne6 c5 Nhf4 b4 Nxe2 Nxe2 Nc7 Nc3 a5 a3 axb4 axb4 f6 b5 Nxb5 Nxb5 cxb5 Qb3 Be6 Qxb5 Rfd8 Rxd8 Rxd8 Rb1 Rd7 Qa5 Bf8 Qa4 Rc7 Rb6 Kf7 Nd2 Rc6 Rxc6 Qd7 Ra6 bxa6 c6 Qd3 Qa5 Be7 Qc7 Qc3 g4 h6 Bxh6 Qxh3 g5 fxg5 Qxe5 Qxh6 Nf3 Qh3 Nd4 Bf6 Qc7 Be7 Qe5 Bc4 c7 Qg4 Kh1 Qh3 Kg1 Qg4 Kh2
Result: 0-1
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 f3 e5 d5 c6 c4 Qb6 Nc3 Be7 Nge2 O-O Ng3 cxd5 Na4 Qc7 cxd5 Bd7 Be3 Rc8 Nc3 Na6 Be2 Bd8 O-O Qa5 Kh1 Nc5 Bd2 Ne8 f4 exf4 Bxf4 Bf6 Bg4 Qd8 Bxd7 Qxd7 Qf3 Rc7 Nf5 Bxc3 bxc3 Na4 Qg3 Nxc3 Rae1 f6 Bxd6 Nxd6 Nxd6 Kh8 e5 Nxd5 e6 Qc6 Rd1 Rf8 Nf7 Kg8 Rxd5 Qxd5 Qxc7 Qxe6 Nd6 Qxa2 Qxb7 Qe2 Qd5 Kh8 Nf7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c6 Nf3 dxc4 a4 Bb4 Bd2 Nf6 e3 c5 Bxc4 Nc6 dxc5 Bxc5 O-O O-O Qe2 e5 Ng5 Bf5 Rad1 Qe7 f3 Bg6 Nge4 Nxe4 Nxe4 Bb4 Bd5 Kh8 Qb5 Rab8 Bxb4 Nxb4 Qc5 Qxc5 Nxc5 Bc2 Rd2 Rbd8 e4 b6 Na6 Nxd5 exd5 Bxa4 Ra1 Bd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 f3 g6 c4 Bg7 Nc3 O-O Be3 e5 Nge2 c6 Qd2 a6 dxe5 dxe5 Qxd8 Rxd8 Na4 Nbd7 c5 Nf8 Nc1 Be6 Nb6 Rab8 Nc4 Nh5 Nb3 Nf4 Nba5 Bxc4 Nxc4 N8e6 Rc1 Nd4 b4 Nfe6 a3 f5 Na5 Bf6 Bc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 Be3 Ng4 Bg5 f6 Bc1 f5 Bg5 Bf6 Bxf6 Qxf6 h3 Nh6 d5 Nd8 c5 Ndf7 cxd6 Nxd6 Nd2 Bd7 Rc1 Nhf7 exf5 Bxf5 Nb3 Ng5 Bg4 b6 Bxf5 gxf5 Qh5 e4 Ne2 f4 h4 Ngf7 Qg4 Kh8 Qxf4 Qe7 Nbd4 Rg8 Rxc7 Qxc7 Qf6 Rg7 Ne6 Ne8 Qxg7 Nxg7 Nxc7 Rc8 Ne6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 c5 d5 Nf6 Nc3 e6 Nf3 exd5 e5 Nfd7 Qxd5 Nb6 Qxd8 Kxd8 Be3 Be6 O-O-O Ke8 Ng5 Nc6 Nb5 Rc8 Be2 Nd4 Bxd4 cxd4 Nxe6 fxe6 Nxd4 Kf7 Bg4 Re8 f4 Bc5 g3 a6 h4 Re7 h5 Rhe8 Nf3 h6 f5 exf5 Bxf5 Kg8 g4 Rc7 Kc2 a5 Rhe1 a4 e6 a3 Bg6 Rec8 Bf7 Kh7 Re5 Be7 Rb5 Na4 Rd7 Nc5 Rxc7 Rxc7 bxa3 Bf6 Rb6 Rc8 Ne5 Ra8 Nd7 Nxd7 exd7 Rd8 Bg6 Kg8 Re6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 c5 e3 d5 Nc3 Nc6 a3 cxd4 exd4 Be7 cxd5 exd5 h3 Ne4 Bd3 Bf5 O-O O-O Na4 Bf6 Be3 Re8 Rc1 Qd6 Nc5 b6 Na6 Rac8 Bb5 Red8 Qa4 Ne7 Nb4 a5 Nd3 Ng6 Ba6 Bd7 Qb3 Rb8 a4 h6 Rc2 Ng5 Nxg5 hxg5 Bb5 Nh4 Bxd7 Qxd7 Rd2 Qf5 f3 Re8 Ne5 Bxe5 dxe5 Rxe5 Bf2 Ng6 Rfd1 Nf4 Bg3 Rbe8 Bxf4 Qxf4 Qxb6 Re1 Kf2 Qh4 g3 Qxh3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 Nd2 h6 Bh4 Bxc3 bxc3 g5 Bg3 d6 e3 Qe7 h4 Rg8 hxg5 hxg5 Rh6 Nbd7 Qa4 Kf8 f3 a6 e4 Kg7 Rh1 Rh8 Rg1 Rh6 Bf2 Rah8 O-O-O Rh1 Bd3 Rxg1 Rxg1 c5 Be3 Ne8 Kc2 f6 Nf1 Kf7 Ng3 Ng7 Kd2 Qd8 Qd1 e5 d5 Rh2 Nf1 Rh8 Bf2 Qc7 Ne3 Qd8 Rh1 Rxh1 Qxh1 Nf8 g3 Bc8 a4 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 a6 Qd2 b5 cxb5 Qa5 b6 Qxb6 Nf3 Bg4 e5 Bxf3 Bxf3 dxe5 d6 Ra7 dxe7 Re8 Nd5 Nxd5 Bxd5 Bf6 Bxf6 Qxf6 Rc1 Qxe7 Qa5 Nd7 O-O Rb8 Rc2 Rb5 Qc3 Rb4 Rfc1 Rd4 Qb3 Kg7 h3 Rc7 Bc4 Rc6 Qc3 Nf6 Qa5 e4 Bf1 e3 Qe1 exf2 Qxf2 Qd6 a3 Ne4 Qe3 a5 Be2 f5 Rc4 Rb6 Bf3 Rd3 Qe1 Nd2 Rxc5 Nxf3 gxf3 Qd4 Kh1 Rxf3 Rc7 Kh6 R1c3 Rxc3 bxc3 Qf4 Re7 Rb2
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Be7 h3 O-O O-O Re8 Re1 Bf8 Bg5 Rxe1 Qxe1 Qe8 Nc3 Qxe1 Rxe1 Be6 Bxf6 gxf6 Ne2 Bd6 Ng3 Bxg3 fxg3 Nc6 c3 Ne7 g4 h6 Kf2 Nc8 Nd2 Nd6 Nf1 f5 gxf5 Nxf5 Bxf5 Bxf5 Re5 Be6 Ng3 Rd8 Nh5 Rd6 Nf4 Rb6 b3 Ra6 Re2 Rd6 g4 Bd7 Re5 c6 Ke3 Kf8 c4 dxc4 bxc4 b6 a4 Be8 a5 Rd7 Kd3 Re7 axb6 axb6 Rxe7 Kxe7 c5 bxc5 dxc5 Bd7 Ke4 f6 Nh5 Kf7 Ng3 Kg6 Kf4 Bc8 h4 Bd7 Ne2 Bc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be2 Be7 O-O O-O f4 a6 Be3 Qc7 Qe1 b5 Bf3 Bb7 a3 Nbd7 Qg3 Rac8 Rae1 Ba8 Kh1 Kh8 Qf2 Qb8 Bc1 Nb6 g4 d5 e5 Ne4 Nxe4 dxe4 Bxe4 Bc5 c3 Bxd4 Qxd4 Rc4 Qd3 Rd8 Qf3 Rxe4 Rxe4 Na4 b4 Rc8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 f4 e6 Nf3 Nge7 d4 cxd4 Nxd4 Nxd4 Qxd4 Nc6 Qf2 d5 Bd3 d4 Nb1 Nb4 O-O Bc5 Nd2 O-O e5 Nxd3 cxd3 Qd5 Ne4 Be7 Qg3 Kh8 Bd2 Bd7 Rf3 f5 exf6 gxf6 a3 a5 Raf1 Rf7 Qe1 b6 f5 exf5 Ng3 Rg8 Qe2 Qe5 Qxe5 fxe5 Nxf5 Bc6 Rg3 Rxg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c5 d5 e6 Nc3 exd5 Nxd5 Nxd5 Qxd5 Be7 e4 O-O Bc4 d6 O-O Nd7 Qd1 Nb6 Be2 d5 exd5 Nxd5 c3 Be6 Qc2 h6 Re1 Bf6 Bf1 Qc7 g3 Rad8 Nd2 Rfe8 Nc4 b5 Ne3 c4 Bg2 Nb6 Bd2 a5 Rad1 Qc5 Bc1 b4 Rxd8 Rxd8 Rd1 Rc8 Bf3 b3 axb3 cxb3 Qe4 a4 Ng4 Bg5 Bxg5 hxg5 Ne3 a3 bxa3 Nc4 Nxc4 Qxc4 Qe3 Qxc3 Rd8 Kh7 Be4 g6 Qxc3 Rxc3 Rd1 b2 Rb1 Bd5
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e6 g3 Bb4 Bg2 O-O O-O a6 Bf4 Re8 Nxc6 bxc6 Bd6 Bxd6 Qxd6 Qe7 Qxe7 Rxe7 e4 Rb8 b3 e5 Rfd1 Kf8 c5 Ne8 f4 Nc7 f5 a5 Kf2 Nb5 Nxb5 Rxb5 Rac1 Rb4 Rc4 Rxc4 bxc4 Ba6 Bf1 g6 Rd6 gxf5 exf5 Re8 f6 Rd8 Bd3 Ke8 h4 e4 Bxe4 Bxc4 Bxh7 Bxa2 h5 Rb8 h6 Rb2 Kg1 Bb1 Bd3 Bxd3 Rxd3 Rb1 Kg2 Rb2 Kh3 Rb1 Rb3
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Bg4 O-O Be7 Bg5 Nbd7 Re1 O-O Nbd2 Re8 c3 c6 Qc2 Bxf3 Nxf3 g6 c4 dxc4 Bxc4 Nb6 Bb3 Nfd5 Bh6 a5 a3 a4 Ba2 Nc7 Re4 Nbd5 Rae1 Bf6 h4 Rxe4 Rxe4 Ne6 h5 Ng7 hxg6 hxg6 Bd2 Nf5 Rg4 Bg7 Bc3 Qd7 Ne5 Qe6 Re4 Nd6 Re2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 Nf3 Nf6 Bg5 O-O e3 Nbd7 Qc2 c6 Rd1 b6 Bd3 h6 Bh4 Bb7 O-O c5 Bg3 cxd4 exd4 dxc4 Bxc4 Bxf3 gxf3 Nh5 d5 Nxg3 hxg3 exd5 Nxd5 Bd6 Ne3 Bc5 Nf5 Rc8 Nxh6 gxh6 Qg6 Kh8 Qxh6 Kg8 Qg6 Kh8 Qh5 Kg7 Qg4 Kh6 Rxd7 Qe8 Kg2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 g6 Nf3 Bg7 Be2 O-O O-O dxc4 Bxc4 Bg4 h3 Bxf3 Qxf3 Nbd7 Rd1 e5 Bb3 Qe7 e4 exd4 Rxd4 Nc5 Bc2 Rfe8 Be3 Rad8 Re1 Rxd4 Bxd4 Nfd7 Bxg7 Kxg7 Qe3 Nb6 f4 Rd8 b3 Ne6 e5 Qc5 Qxc5 Nxc5 Kf2 f6 exf6 Kxf6 g4 Nd3 Bxd3 Rxd3 Ne4 Kg7 Nc5 Rxh3 Re7 Kg8 Rxb7 Nd5 Rxa7 Nxf4 a4 Rc3 Ne4 Rxb3 Nf6 Kf8 Nxh7 Ke8 Nf6 Kd8 a5 Ra3 Ne4 Kc8 Nc5 Nh3 Ke1 Kb8 Rb7 Ka8 Rb6 Rxa5 Rxc6 Nf4 Kd2 Ra3 Rf6 Ra2 Kd1 g5 Ne4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c6 Nf3 Nf6 e3 Nbd7 Qc2 Bd6 Be2 O-O O-O Re8 Rd1 Qe7 e4 Nxe4 Nxe4 dxe4 Qxe4 e5 Bg5 Qf8 Bd3 f5 Qxf5 Nf6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d5 cxd5 Nxd5 Nf3 Nb6 O-O Nc6 e3 e5 dxe5 Qxd1 Rxd1 Nxe5 Nxe5 Bxe5 Nd2 c6 Nf3 Bg7 Nd4 Bg4 f3 Bd7 Bd2 c5 Nb3 Na4 Bc3 Nxc3 bxc3 b6 Rac1 O-O-O Bf1 Be6 c4 f5 Kf2 g5 Be2 f4 gxf4 gxf4 exf4 Bh6 Rxd8 Rxd8 Rg1 Bxf4 Rg7 Rd7 Rxd7 Kxd7 h4 Bf5 Bf1 Kd6 Ke2 Ke5 h5 h6 Kd1 Be6 Bd3 Bf7 Bg6 Bxc4
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Bc4 Nc6 d3 Bc5 Nf3 d6 c3 Nf6 Bb3 O-O h3 Be6 Nbd2 a6 Qe2 Ba7 g4 Nd7 Bc2 d5 Nf1 f6 Ng3 d4 Nf5 Nc5 Rg1 Rf7 cxd4 Bxf5 gxf5 Nxd4 Nxd4 Qxd4 Kf1 Kh8 Rb1 Rd7 Rg3 a5 Be3 Qd6 Rd1 Na6 Bb3 Bd4 Rg4 Nc5 Bc4 Qf8 Rh4 Qe8 Qg4 Bxe3 fxe3 h6 Ke2 c6 a3 b5 Ba2 a4 Rg1 Nb3 Bxb3 axb3 Rc1 Rad8 Rc3 Qg8 Qg6 Rd6 Rg4 R8d7 Rg1 Qd8 Rxb3 c5 Rxb5 Rxd3 Kf3 Rd2 Rxc5 Rxb2 Rgc1 Rb8 a4 Qf8 a5 Ra7 a6 Rba8 Rc6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 d5 d4 Nbd7 Qc2 c6 e3 Bd6 g4 h6 Bd2 a6 e4 dxe4 Nxe4 Nxe4 Qxe4 Qe7 O-O-O c5 Bc3 Nf6 Qe2 b6 dxc5 Bxc5 Ne5 Bb7 f3 Rd8 Bg2 Rxd1 Rxd1 O-O h4 Rd8 Rxd8 Qxd8 Kc2 Nh7 h5 Be7 f4 Qa8 Bxb7 Qxb7 Qf3 Qxf3 Nxf3 Nf8 Ne5 Bd6 b4 Bxe5 Bxe5 Nd7 Bd6 Nf6 a4 Nxg4 Bc7 Kf8 Bxb6 Ke8 b5 axb5 cxb5 Nf6 Bd4 Kd7 a5 Nd5 Bxg7 Nxf4 a6 Nd5 a7
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 d4 cxd4 Nxd4 d5 Nxc6 bxc6 cxd5 cxd5 Nc3 Nf6 Bg5 e6 e4 Be7 exd5 Nxd5 Bxe7 Qxe7 Bb5 Bd7 Bxd7 Qxd7 Nxd5 exd5 O-O O-O Qd4 Rac8 Rac1 Rxc1 Rxc1 Rc8 Rxc8 Qxc8 g3 Qd7 h4 h5 a3 g6 Kf1 a6 Ke2 Qb5 Kd2 Qb3 Ke2 Qc2 Ke3 Qc1 Kd3 a5 Qc3 Qd1 Ke3 Kh7 b4 d4 Qxd4 Qc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 Nc6 Nge2 a6 Qd2 Rb8 h4 h5 Nc1 e5 d5 Nd4 Nb3 c5 dxc6 bxc6 Nxd4 exd4 Bxd4 Be6 Be2 Qa5 Rc1 d5 exd5 cxd5 cxd5 Nxd5 Bxg7 Kxg7 Nxd5 Qxd2 Kxd2 Rxb2 Rc2 Rxc2 Kxc2 Bxd5 a3 Re8 Kd2 Rd8 Ke3 Re8 Kf2 Rb8 Rd1 Bc6 Rc1 Rb6 Rc3 Kf6 Ke3 Ke6 g4 hxg4 fxg4 Bd5 h5 gxh5 gxh5 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8 Kxd8 Nc3 Be7 Bg5 Bxg5 Nxg5 Ke7 Rad1 h6 Nge4 b6 h3 Ba6 Rfe1 Rad8 f4 Rxd1 Rxd1 Ne3 Rc1 Rd8 Kf2 Nd5 Kf3 Nxc3 Nxc3 Ke6 Rd1 Rxd1 Nxd1 Bb7 g3 c5 Ke3 Bg2 Nf2 Kd5 h4 a6 g4 b5 Nd3 Bf1 h5 Bxd3 Kxd3 c4 Ke3 c6 c3 a5 a3 a4 Kf3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 e3 Bb7 Bd3 Ne4 O-O Bxc3 bxc3 f5 Nd2 O-O f3 Ng5 Qe2 d6 e4 Qf6 Ba3 c5 Rae1 Nd7 exf5 exf5 h4 Nf7 Qe6 Qxe6 Rxe6 g6 Rfe1 Rac8 Kf2 Rfd8 g4 fxg4 fxg4 b5 h5 Nf8 Re7 Ba8 Rxa7 b4 Ree7 bxa3 Rxf7 Bc6 Rg7 Kh8 h6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 dxe4 Nxe4 Be7 Nf3 Nf6 Bd3 b6 Nxf6 Bxf6 Be4 c6 O-O Bb7 Qe2 Nd7 Bf4 O-O Rad1 Be7 c4 Nf6 Bc2 Qc8 Rfe1 Re8 Be5 Nd7 Qe4 Nf8 Qg4 f6 Bg3 Qd8 h4 Bd6 h5 Bxg3 Qxg3 Qb8 Qh4 Qc7 c5 Rad8 b4 Bc8 h6 Kh8 hxg7 Qxg7 Re4 Re7 Rg4 Qf7 Rg3 Nd7 Re1 Rg8 Rxg8 Qxg8 d5 cxd5 c6 Nf8 Qxf6 Qg7 Qf4 Rf7 Qe3 Qb2 Ne5 Rg7 Qf4 Kg8 Bd3 Qc3 Re3 Qc1 Kh2 Qd2 b5 d4 Rf3 Qxf4 Rxf4 Rg5 Nf3 Rh5 Kg3 e5 Bc4 Be6 Rxf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 O-O Be2 c5 O-O Nc6 Be3 Bg4 d5 Bxf3 Bxf3 Nb8 Bxc5 Bxc3 Rc1 Bg7 Bd4 Nd7 Bxg7 Kxg7 Qd4 f6 Bg4 b6 f4 Nc5 e5 f5 Bf3 Rc8 Rfd1 h5 h3 Rh8 Kh2 Kf7 d6 e6 a4 a5 Rb1 Rb8 Rb5 Rc8 Rc1 Nd7 Rc6 Rxc6 Bxc6 Nc5 Rb1 g5 fxg5 Qxg5 Rxb6 Ne4 Rb7 Kg6 Bxe4 Qf4 g3 Qxe4 Qxe4 fxe4 Kg2 Rf8 Rb2 Rf5 d7 Rf8 Rd2 Rd8 Kf2 Kf5 Ke3 Kxe5 g4 h4 Rd1 Kf6 Kxe4
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 g3 Nf6 Bg2 d5 Nf3 Be7 O-O O-O c4 dxc4 Qc2 Nbd7 Qxc4 c6 Rd1 Bd6 Nc3 Qc7 Bg5 h6 Bxf6 Nxf6 Rac1 Nd5 Ne4 Nb6 Qc2 Nd7 Nxd6 Qxd6 e4 e5 dxe5 Qe7 Qc3 Re8 h3 Kf8 Rd4 Nxe5 Nxe5 Qxe5 f4 Qe7 b4 Rd8 f5 Rxd4 Qxd4 a5 b5 cxb5 g4 Bd7 Rc7 Rd8 Qb6 Rc8 Qxb7 Rxc7 Qxc7 a4 e5 Be8 Qb8 Qc5 Kh2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 c5 d5 Nf6 Nc3 e6 Nf3 exd5 e5 Nfd7 Bg5 Be7 Bxe7 Qxe7 Nxd5 Qd8 Qc2 Nb6 O-O-O Nxd5 Bxc4 Be6 Bxd5 Bxd5 Qxc5 Na6 Qxd5 Qc8 Kb1 O-O Qe4 h6 Nd4 Nc5 Qc2 Na6 Qxc8 Raxc8 Nf5 Rc7 f4 Kh7 Nd6 Nb8 f5 Nc6 Rhe1 Re7 f6 Rxe5 fxg7 Kxg7 Rxe5 Nxe5 Nxb7 Rb8 Nc5 Rb5 Ne4 Rb4 Nc3 Rf4 Rd2 Ng4 Kc2 Rf6 h3 Ne3 Kd3 Nf1 Re2 Ng3 Re7 Rf2 b4 Rxg2 Rxa7 Nf5 Nd5 Rg3 Ke4 Nd6 Ke5 Nc4 Kd4 Na3 Kc5 Rxh3 b5 Nxb5 Kxb5 Rf3 a4 h5 Rc7 Rf5 Rc5 h4 Ne3
Result: 1-0
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 f3 g6 Be3 Bg7 Qd2 Nc6 Ne2 e5 d5 Ne7 c4 O-O Nbc3 c6 g4 cxd5 cxd5 a6 Ng3 b5 a3 Ne8 Rg1 Bd7 h4 Rc8 h5 Qc7 Bd3 Qb8 Bg5 f6 Be3 Rf7 hxg6 hxg6 Qh2 a5 Nce2 b4 a4 g5 Rh1 Kf8 Nf5 Nxf5 gxf5 Ke7 Nc1 Rf8 Qf2 Qa8 Nb3 Nc7 Nd2 Rh8 Rg1 Rh7 b3 Na6 Bxa6 Qxa6 Nc4 Rxc4 bxc4 Qxc4 Rc1 Qb3 Kf1 Qxa4 Kg2 b3 Rc7 g4 Rb7 Rh3 fxg4 Rh8 Qf3 Qa2 Bf2 a4 Rc1 b2 Rcc7 Rd8 Qd3 Ke8 Rxd7 b1=Q Rxd8 Kxd8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 O-O Nf6 Re1 e6 c3 a6 Bf1 Ne5 d4 cxd4 cxd4 Nxf3 Qxf3 Be7 Nc3 O-O Bf4 Qb6 e5 dxe5 dxe5 Nd5 Nxd5 exd5 Bd3 Be6 Re2 Rfd8 Qg3 Qc6 h4 Kh8 Rd1 Qd7 h5 Rac8 h6 g6 Bg5 Qe8 Qh4 Rd7 Bxe7 Qxe7 Qxe7 Rxe7 Bc2 Kg8 f3 Rd7 Rd4 Rc4 Red2 Rxd4 Rxd4 f6 exf6 Kf7 Kf2 Kxf6 Ke3 Kg5 Rb4 Kxh6 Kd4 Kg7 Rb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bg4 h3 Bxf3 Qxf3 c6 Bd3 e6 O-O Nbd7 Bf4 Be7 Rfe1 Nd5 Nxd5 cxd5 c4 Nf6 c5 O-O a3 Qd8 b4 Ne8 a4 Nc7 b5 Qd7 Rab1 Rfc8 Rec1 Ne8 a5 b6 cxb6 axb6 a6 Nd6 Qe2 Bf8 Rxc8 Rxc8 Rc1 g6 Rxc8 Nxc8 Qc2 Bd6 Bxd6 Nxd6 Qc6 Qxc6 bxc6 Nc8 c7 Kf8 a7 Nxa7 Ba6 Ke7 c8=Q Nxc8 Bxc8 Kd6 Ba6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 b3 d6 Bb2 Nf6 Bb5 Bd7 Bxd7 Qxd7 Qe2 Nc6 O-O Be7 d4 cxd4 Nxd4 O-O Rd1 Nxd4 Bxd4 Qc6 f3 b5 a4 a6 Nd2 Rac8 c4 b4 Nf1 Nd7 Ng3 Bf6 Bxf6 Nxf6 Rd4 Qc5 Qf2 Rfd8 Rad1 g6 R4d2 Qc7 Qe3 Ne8 Ne2 Qc5 Kf2 Kf8 Qxc5 Rxc5 Nf4 a5 Ne2 e5 Rd5 Rdc8 f4 Nf6 fxe5 Nxd5 exd5 dxe5 g4 Rd8 g5 Ke7 Ng3 Rcc8 Ne4 Rd7 Ke3 Rf8 c5 Rc8 h4 h5 Rd2 Kd8 d6 Rc6 Rd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 c6 Nc3 dxc4 e3 b5 a4 b4 Nb1 e6 Bxc4 Be7 O-O O-O Qe2 c5 dxc5 Bb7 Nbd2 Nbd7 Rd1 Qc7 Nb3 Nxc5 Bd2 Nxb3 Bxb3 Rfd8 Be1 Rac8 Nd4 Kf8 f3 Bd6 h3 Qb8 Bf2 Be5 a5 Bd5 Bxd5 Rxd5 Nb3 Qb5 Qxb5 Rxb5 Rac1 Rbb8 e4 Bxb2 Rxc8 Rxc8 Bxa7 h6 Bb6 Ra8 Rd8 Rxd8 Bxd8 Be5 a6 Bb8 Bb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 Bg5 c5 e3 d5 Nbd2 Nbd7 c3 Be7 Bd3 O-O O-O b6 Ne5 Nxe5 dxe5 Nd7 Bf4 f6 Nf3 fxe5 Nxe5 Nxe5 Bxe5 Bf6 Qh5 g6 Bxg6 hxg6 Qxg6 Kh8 f4 Bxe5 fxe5 Bd7 Qh6 Kg8 Rf6 Rxf6 exf6 Qf8 Qg6 Kh8 Qh5 Kg8 Rf1 Qf7 Qg5 Kf8 Rf4 Ke8 Rh4 Bc6 Rh8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 Bg5 c5 e3 d5 Nbd2 Nbd7 c3 Be7 Bd3 b6 O-O Bb7 Qb1 h6 Bf4 Nh5 Be5 cxd4 cxd4 Nhf6 Rc1 Rc8 Rxc8 Qxc8 Qf1 O-O Rc1 Qa8 Rc7 Bd8 Rc2 Be7 a3 Rc8 Qc1 Rxc2 Qxc2 Qc8 Qxc8 Bxc8 Bg3 Bb7 Ne1 Kf8 f3 Ke8 Bb5 Kd8 Kf1 Ne8 e4 Bd6 e5 Be7 f4 Nc7 Bd3 Ba6 Ke2 Bxd3 Kxd3 Nb8 Nc2 Nc6 Bf2 h5 h3 g6 g4 hxg4 hxg4 Kd7 Nf3 Na5 Ne3 b5 Bh4 Bxh4 Nxh4 Nc4 Nd1 a5 Kc3 Ke7 Nf3 Kd7 Ne1 Kc6 Nc2 Kb7 Nf2 Na6 Nd3 Kc6 Nde1 Kd7 Nf3 Nc7 b3 Nb6 Nce1 Ke7 Nd3 Nd7 Nh4 Nb8 f5 gxf5 gxf5 Nc6 f6 Ke8 b4 a4 Nc5 Nd8 Nf3 Na8 Nd2 Nb6 Kd3 Nc6 Nb7 Nc8 Nb1 Nd8 Nxd8 Kxd8 Nc3 Na7 Ke3 Ke8 Kf4 Kf8 Kg5 Kg8 Kh6 Kh8 Nxd5 Nc6 Nc3 Nxd4 Kg5 Nc6 Kf4 Na7 Ne4 Kh7 Nd6 Kg6 Ke4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 Bxc6 bxc6 O-O Bg7 Re1 e5 c3 Nh6 d4 cxd4 cxd4 exd4 Nxd4 O-O Nc3 Re8 Nb3 d5 Bg5 Be6 Nd4 Qb6 Nxe6 fxe6 Qc2 dxe4 Rxe4 Nf5 Rd1 h6 Bc1 Rad8 Rxd8 Rxd8 h4 Rd4 Rxe6 Rd6 Qe4 Rd4 Qxc6 Qd8 Re8 Kh7 Qe6 g5 Qg8 Kg6 Re6
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 Bg5 Nf6 Bxf6 exf6 e3 d5 c4 Bb4 Nc3 Bxc3 bxc3 O-O Bd3 dxc4 Bxc4 Kh8 Ne2 Nc6 Qc2 Ne7 O-O c6 a4 g5 Nc1 Qc7 Nd3 Ng6 Rfe1 b6 a5 Bd7 Qb2 b5 Ba2 Rae8 Qb4 Re4 f3 Rh4 g3 Rh3 f4 Qb8 Re2 Kg7 c4 Re8 Rae1 Rc8 Nf2 Rh5 Rc2 Re8 Rcc1 Kh6 cxb5 gxf4 exf4 cxb5 Nd3 Re4 Bd5 Qe8 Qd6 Nxf4 Qxf4 Rxf4 Rxe8 Rxd4 Re7 Rxd5 Nf4 Rd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8 Rc1 Bg4 h3 Be6 Nxe6 fxe6 f4 Bf8 Bf2 Rc8 a3 Kh8 Qd3 Qd7 Rfd1 Qf7 e4 dxe4 Nxe4 e5 Nxf6 Qxf6 fxe5 Nxe5 Qe4 Rxc1 Rxc1 Rd8 Rf1 Nd3 Bxa7 Qxb2 Rxf8 Rxf8 Qxd3 b6 Qe3 Rc8 Bxb6 Qc1 Qxc1 Rxc1 Kh2
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e6 e4 c5 e5 Ng8 Nf3 d6 exd6 Bxd6 d4 cxd4 Nxd4 Bc5 Be3 e5 Ndb5 Bxe3 fxe3 Qxd1 Rxd1 Na6 Nd5 Kf8 Be2 Be6 Nbc7 Nxc7 Nxc7 Rb8 Nxe6 fxe6 Rd7 Ne7 c5 a6 Bg4 Nf5 Ke2 Ke8 Rhd1 Nd4 R1xd4 exd4 Bxe6 d3 Rxd3 Ke7 Bd5 Rhc8 b4 a5 a3 Rc7 Kd2 axb4 axb4 Ra8 Kc2 Kf6 Kb3 Ke5 Bf3 Rf8 Kc4 Rf6 Rd5 Ke6 Kb5 Ke7 h4 Re6 e4 Rh6 h5 b6 Rg5 Kf8 e5 bxc5 bxc5 Ke8 c6 Re6 Kb6 Kd8 Bd5 Ree7 Rf5 Ke8 e6 Rc8 Rf7 Rb8 Kc5 Rc8 Rf3 Rb8 Ra3 Kf8 g4 Ke8 g5 Kf8 h6 g6 Kd6 Rd8 Ke5 Rb8 Rc3 Rc8 Rb3 Rcc7 Rb7 Rc8 Kd6 Rd8 Rd7 Rdxd7 cxd7 Rxd7 Kxd7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 cxb5 a6 f3 e6 e4 Bb7 Nc3 d6 bxa6 Nxa6 Bg5 Nc7 Bc4 Be7 Nh3 e5 Be3 Qc8 O-O O-O Qe2 Ba6 Bxa6 Qxa6 Qxa6 Rxa6 Nf2 Rb8 Rfb1 Nd7 b3 Nb5 Nxb5 Rxb5 Nd1 Rab6 Bd2 Rb8 Kf1 Kf8 Ke2 Bd8 Ne3 g6 Nc4 Bc7 g4 R5b7 h4 Ra8 h5 Rba7 a4 Nb6 Na3 c4 Nb5 Rb7 Be3 Bb8 Bxb6 Rxb6 bxc4 Rba6 Rb4 Ke7 hxg6 hxg6 Kd3 Kd7 Kc2 Ba7 Nxa7 R6xa7 Kb3 Rh8 Rc1 f5 gxf5 gxf5 Rg1 Rh7 exf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 d5 a3 Bxc3 bxc3 O-O cxd5 exd5 Bd3 c5 Ne2 Nc6 f3 c4 Bc2 a6 O-O Re8 Ng3 Qa5 Bd2 b5 e4 Qb6 Kh1 dxe4 fxe4 Bg4 Qc1 Rad8 Bg5 Ne7 h3 Nh5 Nxh5 Bxh5 Bxe7 Rxe7 Qg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 Nc6 d5 Ne7 g4 c6 Ng3 cxd5 cxd5 a6 h4 b5 h5 b4 Na4 Rb8 b3 a5 Qd2 Nd7 Bd3 f5 gxf5 gxf5 h6 Bf6 Nh5 f4 Ba7 Ra8 Nxf6 Rxf6 Bf2 Qf8 Nb2 Rxh6 Ke2 Kh8 Qe1 Ng8 Rxh6 Nxh6 a3 Nc5 axb4 Nxb3 Ra3 a4 Rxa4 Bd7 Rxa8 Qxa8 Qc3 Qa2 Bc4 Ba4 b5 Nf7 Bh4 Qa3 Bxb3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Ne1 Ne8 Be3 f5 f3 f4 Bf2 h5 c5 g5 a4 Ng6 a5 Bh6 b4 Kh7 cxd6 cxd6 Nb5 g4 fxg4 hxg4 Bxg4 Bxg4 Qxg4 Nf6 Qf3 Qd7 Nxa7 Ng4 b5 Nxf2 Qxf2 f3 Nxf3 Qg4 b6 Qxe4 Rfe1 Qxd5 Qc2 Rae8 Rad1 Qxa5 Qc7 Re7 Qxd6 e4 Nh4 Rg7 Nxg6 Rxg6 Qe7 Bg7 Qh4 Kg8 Qxe4 Rxb6 Qd5 Qxd5 Rxd5 Rb2 Rf1 Bh6 Rxf8 Kxf8 Rf5 Ke7 Nb5 Be3 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 Bb4 Qb3 a5 a3 Bxc3 Qxc3 Nf6 d4 O-O Bg5 d6 e3 Qe7 Bd3 e5 Ne2 Nc6 f3 h6 Bh4 a4 O-O Bd7 Rae1 Rfe8 Qc2 Na5 Nc3 g5 Bg3 Nb3 f4 e4 Be2 Kg7 fxg5 hxg5 Rf2 Rh8 Ref1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5 e4 c6 b3 Qb6 Rb1 exd4 Nxd4 Nxe4 Nxe4 Bxd4 Bb2 Bxb2 Rxb2 d5 cxd5 cxd5 Qxd5 Nf6 Nxf6 Qxf6 Rd2 a5 Re1 a4 Qd4 Qxd4 Rxd4 axb3 axb3 Be6 Bxb7 Rab8 Bd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7 Nxd7 Bxd7 O-O Qh4 c4 O-O-O c5 g5 Be3 Re8 Nd2 Bg7 Nf3 Qh5 Nxg5 Qxd1 Rfxd1 Nxg5 Bxg5 Bg4 Rd2 Bxd4 c6 Be5 Bb5 b6 Bh4 Rhg8 Ba6 Kb8 Bg3 Bxg3 hxg3 Rd8 Rd4 Rd6 Rc1 Bc8 Bd3 Rg4 Rxg4 Bxg4 f3 Be6 b4 d4 a4 a5 b5 Bb3 Ra1 Kc8 Kf2 h6 Ke2 Kd8 Kd2 Ke7 Bc2 Bc4 Rh1 d3 Bd1 Kf8 Rh4 Ba2 Re4 Rg6 g4 h5 f4 hxg4 f5 Rd6 Rxg4 Bd5 Rh4 Kg7 g3 Kf6 Rh8 Ke5 Rc8 Kd4 Rxc7 Rh6 Rd7 Rh2 Kc1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Qb6 a3 Nc6 Nb3 Qc7 Bd3 g6 O-O Bg7 Bd2 O-O Nd1 e5 Ne3 exf4 Qxf4 Be6 Qh4 Qd8 Rae1 Nd7 Qxd8 Raxd8 Na5 Nxa5 Bxa5 Rc8 c3 Bh6 Nd5 Bxd5 exd5 Ne5 Be2 Bd2 Rd1 Be3 Kh1 f5 g3 g5 Bb4 Rf6 b3 f4 gxf4 gxf4 c4 f3 Bxf3 Nxf3 Bxd6 Rxd6 Rxf3 Bh6 Re1 Rg6 Re4 b5 c5 Rxc5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 d5 Nf3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Qc2 Nc6 a3 Qa5 O-O-O Ne4 Nb5 a6 Nc7 e5 Rxd5 f5 Rxe5 Nxe5 Bxe5 Ra7 Nd5 b6 Bd3 Bd7 b4 Qxa3 Bb2 Qa4 bxc5 bxc5 Ne5 Qxc2 Bxc2 Be6 Nf4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Bd3 Nc6 Be3 Nf6 O-O Bd6 Nxc6 bxc6 f4 e5 f5 Be7 Na4 d5 Bb6 Qb8 Qe2 c5 Bxc5 Bxc5 Nxc5 Qb6 Qf2 Ng4 Na4 Qxf2 Rxf2 Nxf2 Kxf2 d4 b4 Bd7 Nc5 Bb5 a4 Bxd3 cxd3 Ke7 Rc1 Rhc8 Rc4 Rc6 a5 Rb8 Ke2 Kd6 h4 h5 Kf3 Rb5 g4 g6 g5 Rb8 f6 Rb5 Ke2 Rcxc5 bxc5 Kc6 Ra4 Rxc5 Ra2 Rb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2 cxd4 exd4 O-O a3 Be7 Nf4 d6 Be2 Re8 O-O Bf8 Be3 g6 Rc1 a6 c5 Bg7 Na4 Nc6 Nb6 Rb8 b4 d5 Qb3 g5 Nh3 h6 Rcd1 e5 dxe5 Nxe5 Nxc8 Qxc8 Rfe1 Re7 Bf1 Qc7 Bd4 Rbe8 Bxe5 Rxe5 Rxe5 Qxe5 f3 g4 Nf2 g3 hxg3 Qxg3 Qd3 h5 Qf5 Re6 Nh1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 O-O-O a6 Kb1 b5 Nc1 exd4 Bxd4 Re8 Bxf6 Qxf6 Qxd6 Qxd6 Rxd6 Ne5 f4 Ng4 e5 Nf2 Rg1 Bf5 Ka1 b4 Na4 f6 e6 Rxe6 Rxe6 Bxe6 Be2 f5 Nb3 Bf7 Na5 Rd8 Rf1 Ng4 Rd1 Rxd1 Bxd1 Ne3 Bf3 Nxc4 Nxc6 a5 Nd8 Nd2 Bc6 Bh6 g3 Nf1 Nb6 Nxh2 Nd7 Bg7 Ne5 Bxe5 fxe5 Kf8 e6 Be8 Bxe8 Kxe8 Nc6 Nf1
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 c4 c6 d4 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O O-O Qe2 Nbd7 Ne5 Re8 Rd1 Qc7 Nxd7 Qxd7 f3 Nd5 Na2 Bf8 e4 Bg6 Qe1 f5 exd5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 f3 Ng8 Qc2 Ngf6 Nb5 axb4 axb4 Nh5 g3 Ndf6 c5 Bd7 Rb3 Bh6 Rc3 Bf4 cxd6 Nxg3 hxg3 Nh5 gxf4 Nxf4 Bc4 Nh3 Kh1 Qh4 Nb3 fxe4 Qh2 Rf5 f4 Rh5 Qg3 Qxg3 Rxg3 exf4 Bb2 Kg8 dxc7 Bxb5 Bxb5 fxg3 Kg2 Ng5 d6 Rh2 Kxg3 Rxb2 Bc4 Kg7 d7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Nb6 Nd2 Qe6 b3 Bb4 Bb2 O-O O-O-O d5 exd6 Qxd6 Nf3 Qh6 Qe3 Qxe3 fxe3 Re8 e4 Nd7 e5 a5 Bd3 Nc5 Bc2 a4 Nd4 axb3 axb3 Ra6 h3 Ne6 Rhf1 Nxd4 Rxd4 g6 Be4 Be6 Kc2 Be7 Ra1 Rxa1 Bxa1 Ra8 Bb2 Ra2 c5 Ra5 b4 Ra6 Bc3 f6 Rd1 fxe5 Ra1 Rxa1 Bxa1 Bf6 Bxc6 Bf5 Kb3 Kf7 b5 Ke7 Kc4 Be6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 Nf6 e5 Nd5 Nf3 Nc6 Bc4 Nb6 Bb3 d5 exd6 exd6 d4 Bg4 h3 Bh5 Be3 Be7 dxc5 dxc5 Na3 O-O O-O Qc8 Bf4 Rd8 Qe2 Bf8 Rfe1 Qf5 Bg3 Bxf3 Qxf3 Qxf3 gxf3 Rd7 Rad1 Rad8 Rxd7 Rxd7 Re8 g6 Nc4 Nxc4 Bxc4 Kg7 Bf4 Bd6 Be3 a6 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 e4 c6 h3 Qb6 c5 dxc5 dxe5 Ne8 Na4 Qa6 Bg5 b5 Nc3 Nc7 Be7 Re8 Bd6 Ne6 a4 b4 Ne2 Qa5 Nd2 Ba6 f4 c4 Kh2 Rad8 Qc2 Nb6 Rfd1 Bf8 Nf3 c3 Ned4 Nxd4 Nxd4 cxb2 Qxb2 Nc4 Qb3 Qb6 a5 Qb7 Bxf8 Kxf8 e6 c5 e5 Qc7 exf7 Qxf7 Nc6 Rxd1 Rxd1 Nxe5 Bd5 Bc4 Qc2 Ng4 hxg4 Re2 Qxe2 Bxe2 Bxf7 Bxd1 Bc4 b3 Nxa7 b2 Ba2 Be2 Kg2 Bd3 Kf3 Ke7 Ke3 b1=Q Bxb1 Bxb1 Nb5 Kd7 a6 Kc6 f5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 a6 Qd2 b5 h4 Bb7 h5 Be7 dxc5 Nxc5 Bd3 Rc8 Bxc5 Bxc5 f5 Nb4 Qf4 Nxd3 cxd3 exf5 Qxf5 Rc7 d4 Bc8 Qf4 Be7 O-O h6 Rac1 Be6 Ne2 O-O Qg3 Rc4 b3 Rxc1 Rxc1 Ba3 Rf1 Qd7 Nh4 Rc8 Nf4 Rc1 Nxe6 Qxe6 Qf3 Rxf1 Kxf1 g6 hxg6 fxg6 g3 Bb2 Qf2 Bc3 Qe3 b4 Qxh6 Qh3 Ke2 Qg4 Kf1 Qh3 Ke2 Qg4 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c6 e3 Nf6 Nf3 Nbd7 Qc2 Bd6 g4 dxc4 e4 e5 g5 exd4 Nxd4 Ng4 h3 Nge5 Be3 Nc5 O-O-O Ned3 Kb1 Qe7 Rg1 g6 Bg2 O-O Ka1 Bf4 Bxf4 Nxf4 h4 Rd8 Qd2 Ncd3 Qe3 Bg4 Rd2 Qe5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 Bxc6 bxc6 O-O Bg7 Re1 Nf6 e5 Nd5 c4 Nc7 d4 cxd4 Qxd4 O-O Qh4 d6 Bh6 Ne6 Nc3 f6 Bxg7 Kxg7 exf6 Rxf6 Ng5 Nxg5 Qxg5 Qf8 Re2 Ba6 b3 e5 Rd1 Rf4 Red2 Rd8 Qg3 c5 Ne2 Rf5 Qe3 Bb7 f3 h5 Rd3 Rd7 Nc3 Bc6 Qd2 Rf6 Nd5 Re6 h3 Qf5 Re1 Qf7 a3 Rd8 b4 cxb4 axb4 Qb7 b5 Bxd5 Rxd5 Qb6 Kh2 Rc8 f4 Rce8 fxe5 dxe5 Rd7 R8e7 Qd5 Kh6 Qd2 Kh7 Rf1 e4 Rf7 Rxf7 Rxf7 Kg8 Qd7 Qb8 Kg1 Qe5 Rg7 Qxg7 Qxe6 Kh7 Qxe4 Qc3 Qe7 Kh6 Qxa7 Qc1 Kh2 Qxc4 Qb8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5 Qd2 f5 O-O-O Nd7 Bd3 Nc5 Bc2 a6 Nge2 b5 b4 Nd7 cxb5 axb5 Nxb5 Rxa2 Nec3 Ra8 Kb2 Ndf6 Na7 fxe4 Nc6 Qd7 g4 Nf4 g5 N6xd5 Nxd5 Nd3 Bxd3 exd3 Nce7 Kh8 Nxc8 e4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bc4 e6 Qe2 Nb6 Bb3 h6 N5f3 c5 Bf4 Bd6 Bg3 Qe7 dxc5 Bxc5 Ne5 Bd7 Ngf3 Nh5 O-O-O Nxg3 hxg3 O-O-O Rh5 Be8 Rxd8 Kxd8 Qd2 Bd6 Nd3 Qc7 g4 Kc8 g5 Bf8 Rh4 Kb8 a4 Be7 a5 Nd5 Kb1 Bd8 a6 Qa5 Qe2 Nb6 axb7 Bxg5 Nxg5 Qxg5 Rh5 Qf6 Ra5 Bc6 Nc5 Bxb7 Nxb7 Kxb7 Qa6 Kc6 Ba4 Kd6 Qd3 Nd5 Qg3 Qe5 Qa3 Kc7 Qc5 Kd8 Rxa7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 f3 Ng8 Qc2 Ngf6 Bd3 f4 Nb5 b6 c5 dxc5 bxa5 Rxa5 Nc4 Ra8 a4 Ne8 a5 Ba6 Bd2 Bxb5 Rxb5 Nd6 Rb2 Qe7 Nxd6 Qxd6 Bb5 bxa5 Bc6 Rab8 Ra2 Rf7 Rxa5 Bf8 Bc1 g5 Bb2 Qf6 Bxd7 Rxd7 Rfa1 Bd6 Ba3 g4 Bxc5 Rg8 Ra8 Rg7 Kh1 Qg6 Bxd6 cxd6 Rc8 gxf3 Rxg8 Rxg8 gxf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7 h4 Nbc6 h5 Qa5 Bd2 cxd4 cxd4 Qa4 Nf3 Nxd4 Bd3 Nec6 Kf1 Nxf3 Qxf3 b6 h6 Ba6 hxg7 Rg8 Bxa6 Qxa6 Kg1 Rxg7 Qf6 Rg8 Rxh7 Qb7 Bg5 Nd4 c4 Ne2 Kh2 Nc3 Rh8 Rxh8 Qxh8 Kd7 Qh7 Rf8 Bh6 Re8 Qxf7 Re7 Qg6 Qb8 cxd5 Nxd5 Rd1 Qxe5 f4 Qh8 f5 Qe5 Kh1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 Be3 c6 Qd2 exd4 Nxd4 Re8 f3 d5 exd5 cxd5 O-O Nc6 c5 Rxe3 Qxe3 Qf8 Ncb5 Qxc5 Rac1 Qb6 Qf2 Bd7 Rfd1 Re8 Bf1 Bh6 Rc3 Nb4 Nc2 Qxf2 Kxf2 Nxc2 Rxc2 Be3 Ke1 Bf4 Kf2 Be3 Ke1 Bg1 Kd2 Bxh2 Re1 Rd8 Kd1 Bf4 Nd4 a6 Bd3 h5 Rc3 h4 Bc2 Bg3 Re2 Kg7 Rb3 Rb8 Rd2 b5 Ra3 Rb6 b4 Bd6 Rb3 Nh5 Ne2 Be6 a3 g5 Nd4 Kf6 Nxe6 fxe6 Bd3 Nf4 Rc3 d4 Rc8 Nd5 Ke2 Bf4 Rdc2 Ne3 Kf2 Bg3 Kg1 Nxc2 Rxc2 g4 fxg4 Kg5 Kf1 e5 Ke2 Kxg4 Rc8 Rf6 Be4 Rf2 Kd3 Bf4 Rg8 Kh5 Bd5 Bg5 Ke4 Rxg2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Bc4 Ngf6 Ng5 e6 Qe2 Nb6 Bb3 a5 c3 a4 Bc2 a3 b3 Nbd5 Bd2 Bd6 N1f3 Nf4 Qf1 h6 Ne4 Nxe4 Bxe4 O-O g3 Nd5 Qe2 c5 dxc5 Bxc5 Ne5 Qc7 O-O Be3 fxe3 Qxe5 Qd3 Ra6 c4 Nf6 Bg2 Rd6 Qc3 Qg5 Qa5 e5 Bb4 Ra6 Qb5 Bd7 Qxb7 Qxe3 Kh1 Ra7 Bd2 Qd4 Bc3 Qc5 Qb4 Qc7 c5 Bc6 Qb6 Bxg2 Kxg2 Nd5 Qxc7 Rxc7 Bxe5 Rxc5 Bd6 Rc2 Rf2 Rfc8 Bxa3 Ne3 Kg1 Ra8 Bb2 Rxf2 Kxf2 Nc2 Rc1 Nb4 Rb1 Rxa2 Ke3 f6 Kd4 Nc2 Kc3 Na3 Bxa3 Rxa3 Kb2 Ra7 b4 Kf7 b5 Ke6 Kb3 Kd7 h4 h5 Rc1 Rc7 Rd1 Kc8 Rd5 Re7 Kc4 Re3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 Na6 h4 h6 Be3 e5 d5 Nc5 Qc2 c6 h5 g5 f3 a5 g4 Bd7 Nh3 a4 Qd2 cxd5 cxd5 Qa5 Nb1 Nfxe4 fxe4 Nxe4 Qxa5 Rxa5 Nc3 Ng3 Rg1 Nxe2 Kxe2 e4 Rac1 f5 gxf5 Rxf5 Nf2 Be8 Rh1 Bb5 Nxb5 Rxb5 Rc8 Kh7 Rd1 Rxb2 Rd2 a3 Rc7 Rxd5 Nxe4 Kg8 Rcc2 b5 Rxd5 Rxc2 Rd2 Rb2 Kd3 d5 Nc5 Rb4 Ne6 Rh4 Nxg7 Kxg7 Bd4 Kg8 Be5 Kf7 Re2 Rxh5 Kd4 Ke6 Bb8 Kf5 Re3 Rh1 Rf3 Kg4 Rg3 Kf5 Rf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 O-O O-O Bb6 Nc2 d6 Bxb6 axb6 f4 g5 f5 Ne5 Be2 Bd7 c4 g4 Nc3 h5 Qd2 Kh8 Qf4 Bc6 Ne3 Nd7 Bxg4 hxg4 Nxg4 Qh4 Rf3 Ng6 Qe3 Qxg4 Qh6 Kg8 Rh3 Qxh3 gxh3 Nge5 f6 Nxf6 Qxf6 Rae8 Kh1 Ng6 h4 Re6 Qg5 Rfe8 h5 Re5 Qh6 Rxe4 Nxe4 Rxe4 Kg1 Ne5 Qg5 Kh7 Qf5 Kh6 Rf1 Re2 Qf6 Kh7 Qg5 Be4 h6 Bg6 h4 Re4 h5 Rg4 Qxg4 Nxg4 hxg6 fxg6 Rf7 Kxh6 Rxc7 Ne5 Rxb7 Nxc4 b3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5 Bxf7 Rxf7 fxg4 Rxf1 Kxf1 Qd6 e5 Qd5 g5 Qe4 Bf2 Rf8 Ng1 Nc4 Nf3 Ne3 Bxe3 Qxe3 Qb3 Kh8 Re1 Qxg5 Qxb7 Qd2 dxc5 Qxc3 Qd5 Bh6 c6 Bf4 Re4 Qc1 Re1 Qc3 Re4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Nf6 Nxc6 bxc6 e5 Nd5 Ne4 Qc7 f4 Qb6 c4 Ne3 Qd3 Nf5 g4 Nd4 Bg2 h5 g5 Bb7 Be3 Nc2 Kf2 Nxe3 Qxe3 Qxb2 Kg3 Ba6 Rab1 h4 Kh3 Qc2 Rhc1 Qa4 Qc3 Ba3 Rc2 Be7 Rb3 O-O Nf6 gxf6 gxf6 Bc5 Be4 Rfb8 Kxh4 Kf8 Rg2 Qxc4 Qxc4 Bxc4 Bh7 Bf2 Kh5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5 e4 c6 Be3 Ng4 Bg5 Qb6 h3 exd4 Na4 Qa6 hxg4 b5 Be7 Re8 Bxd6 bxa4 e5 c5 b4 cxb4 Qxd4 Bb7 Rae1 Rac8 c5 Qb5 Qf4 Bxf3 Bxf3 Nxc5 Be2 Qb6 Rc1 Ne6 Qxb4 Nd4 Bc4 Bxe5 Qxb6 axb6 Bxe5 Nf3 Kg2 Nxe5 Bb5 Rxc1 Rxc1 Rd8 Bxa4 Rd2 g5 Rxa2 Rc8 Kg7 Bb3 Rb2 Rc3 b5 Re3 Ng4 Rf3 f5 gxf6 Nxf6 Rd3 Ne4 Rd7 Kh6 Bg8 Rxf2 Kg1 Rb2 Rxh7 Kg5 Rb7 Kg4 Kf1 g5 Bh7 Nc3 Rc7 b4 Rc5 Rd2 Ke1 Re2 Kf1 Rd2 Ke1 Rd5 Rxd5 Nxd5 Kd2 Kxg3 Be4 Nb6 Kd3 g4 Bc6 Kf2 Kc2 Nc4 Bd7 g3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Bxf6 gxf6 Nd5 f5 Bd3 Be6 Qh5 Rg8 g3 Rc8 c3 Rg6 Nc2 Rh6 Qe2 Bxd5 exd5 Ne7 f3 Bg7 Ne3 f4 gxf4 Ng6 O-O-O Nxf4 Qc2 Qb6 Nf5 Rg6 Be4 b4 c4 Kf8 Rd2 Rc5 Ng3 Qc7 Kb1 Bh6 Qb3 Rg7 Nf5 Rg6 Qxb4 Rxc4 Qxc4 Qxc4 Rc2 Qxc2 Kxc2 Rf6 Ne3 Ne2 Ng4 Nd4 Kd3 Rf4 Nxh6 Kg7 Nxf7 Rxf7 Rf1 Rb7 b3 a5 f4 Rb4 fxe5 dxe5 Rc1 Nb5 Rc6 Kf7 Ke3 Ke7 Bxh7 Rh4 Bd3 Nd6 Rc7 Kd8 Rh7 Rd4 h4 Rg4 h5 Rg3 Kd2 Rg2 Ke1 e4 Be2 Nf5 Rf7 Nd4 Bf1 Rh2 Rf4 Nc2 Kd1 Ne3 Ke1 Nc2 Kd1 Ne3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 e5 Bg2 d6 d3 f5 e4 Nf6 Nge2 O-O h3 c6 Be3 Be6 Qd2 fxe4 Nxe4 Nxe4 Bxe4 d5 cxd5 cxd5 Bg2 Nc6 O-O Qd7 Kh2 d4 Bg5 Bd5 Be4 Rf7 Kg2 Qe6 Bxd5 Qxd5 f3 a5 h4 a4 Ng1 a3 Rfb1 Bf8 bxa3 Rxa3 Rb2 b5 Rab1 b4 Rc2 Rc3 Rbb2 Na5 Qd1 Nb7 Bd2 Rxd3 Qe2 Rdxf3 Nxf3 d3 Qe3 dxc2 Rxc2 Nc5 Bxb4 Nd3 Bd2 Bc5 Rxc5 Nxc5 Bc3 Nd3 a4 Rc7 Bd2 Nb2 Bc3 Nc4 Qe2 Rf7 a5 e4 Ne1 e3 Kg1 Nd2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Ne1 Ne8 Be3 f5 f3 f4 Bf2 h5 c5 g5 cxd6 cxd6 a4 Ng6 Nb5 a6 Na3 Nf6 Nc4 g4 Qb3 g3 hxg3 fxg3 Bxg3 h4 Bh2 Nh5 Qb6 Qg5 Nd3 Ng3 Rf2 Nxe2 Rxe2 Rxf3 Nxd6 Rxd3 Nxc8 Kh7 Nd6 Nf4 Bxf4 exf4 Nf5 Rg8 Qxb7 f3 Rf2 Rd2 Qa7 Kh8 Qe3 Qxe3 Nxe3 Bd4 Rxd2 Bxe3 Kh2 Bxd2 gxf3 Rg3 Rd1 Bf4 d6 Rxf3 Kg2 Rg3 Kh1 Rg8 d7 Rd8 Kg2 Kg7 Rd5 Bc7 b4 Kf7 a5 Rg8 Kh3 Bd8 b5 axb5 Rxb5 Rg3 Kh2 Ra3 Rh5 Ke6 a6 Kxd7 Rh6 Bg5 Rh5 Be7 Rh6 Ra4 e5 Ra5 Rb6 Rxe5 a7 Ra5 Rb7 Ke6 Kh3 Ra4
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4 Kh8 Qc2 axb4 axb4 Nf6 f3 g5 c5 Ng6 Nc4 Nf4 cxd6 cxd6 Be3 g4 Bb6 Qe7 exf5 Nxe2 Nxe2 gxf3 Rxf3 Nxd5 Qd2 e4 Rff1 Rxf5 Rxf5 Nxb6 Qxd6 Qxd6 Nxd6 Bxf5 Nxf5 Nd5 Nxg7 Kxg7 Rd1 e3 g3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bc4 Qb6 Nde2 e6 O-O a6 Bb3 Be7 Bg5 Qc7 Ng3 b5 Kh1 h5 Bxf6 gxf6 Nxh5 Bb7 Ng3 O-O-O a4 b4 Nce2 Kb8 Rc1 Qa5 c3 Rh4 Ng1 Ne5 Qd2 d5 exd5 Nc4 Qe2 Bxd5 Rfd1 Rc8 Nf3 Rf4 Rd4 Bd6 Rc2 bxc3 bxc3 f5 Qd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 e5 dxe5 d4 Ne4 Qa5 Bd2 Qxe5 Ng3 Qd6 Nf3 Nf6 Qc2 Be7 O-O-O O-O e3 dxe3 fxe3 Qc7 Bc3 Bg4 Bd3 Nbd7 Bf5 Bxf5 Nxf5 Rfe8 Nxg7 Kxg7 Qf5 Nf8 h4 h6 g4 Qc8 Qxc8 Raxc8 g5 Nh7 e4 Rcd8 Rdf1 Kf8 gxf6 Bxf6 e5 Bg7 Rhg1 c5 Kc2 Re6 Rg4 Bh8 b4 b6 bxc5 bxc5 Rb1 Ra6 Rb2 Bg7 Rb7 Rxa2 Kb3 Ra6 e6 Rxe6 Rxg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 a6 Qd2 Nc6 Nge2 Rb8 h4 b5 h5 e5 d5 Na5 Ng3 bxc4 O-O-O Rb4 Bh6 Bxh6 Qxh6 Qe7 Be2 Bd7 Nf1 Rfb8 Rd2 c5 Bd1 Ne8 hxg6 fxg6 g4 Qg7 g5 Qxh6 Rxh6 Ng7 f4 exf4 Rdh2 Re8 Nd2 Reb8 Rxh7 Rxb2 R2h4 R2b7 Rh8 Kf7 Rxf4 Ke7 Rh7 Rg8 Rf6 Be8 e5 Rb6 Nde4 Nb7 exd6 Nxd6 Re6 Kd7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nc6 Nc3 Nf6 g3 d6 Bg2 g6 d4 cxd4 Nxd4 Bd7 e4 Bg7 Nxc6 Bxc6 Be3 O-O O-O a5 Rc1 a4 Qe2 Qa5 Rfd1 Rfc8 c5 Ne8 cxd6 Nxd6 Nd5 Bxd5 Rxd5 Rxc1 Bxc1 Qc7 Bf4 Qc4 Qxc4 Nxc4 b3 axb3 axb3 Na5 b4 Nc6 b5 Nd4 Bf1 h5 Kg2 Nc2 Bd2 e6 Rc5 Nd4 Be3 b6 Rc7 Ra1 Rc8 Kh7 Rc7 Kg8 Rc4 Rd1 Rc8 Kh7 Rd8 Be5 h3 Kg7 Bc4 Kf6 Rd7 g5 Be2 Nxe2 Rxd1 Nc3 Rd8 Nxe4 Bxb6 Nc3 Rg8 Nxb5 Bd8
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 b4 Bb7 Bb2 d6 e3 Nbd7 d4 Ne4 Qc2 c5 Bd3 f5 O-O Qe8 Nd2 Nxd2 Qxd2 Qh5 Be2 Qg5 f3 Rad8 Rad1 e5 dxe5 dxe5 b5 f4 Qd6 fxe3 Qc7 Bc8 Rd5 Rde8 Rxd7 Bxd7 Qxd7 Rd8 Qe6 Kh8 Qxe5 Qxe5 Bxe5 Rd2 Re1 Ra2 f4 Rd8 h4 Rdd2 Bf3 Rxa3 h5 h6 Kh2
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 c5 Bg2 Bg7 d4 O-O O-O Na6 Nc3 d6 b3 Bg4 Bb2 Qd7 Re1 Rab8 Qd2 Bxf3 exf3 Rfe8 f4 cxd4 Qxd4 Nc5 Rad1 b6 Nd5 Ne6 Qd2 Nxd5 cxd5 Nc5 Bxg7 Kxg7 b4 Na4 Qd4 Kg8 Rc1 a5 a3 axb4 axb4 b5 Rc6 Rb7 f5 Qxf5 Rxd6 Reb8 Rc6 Rd8 h4 Qd7 h5 Qe8 Bh3 Qf8 Kg2 Qg7 Re5 e6 Qe3 Nb2 h6 Qf6 dxe6 Nd1 exf7 Qxf7 Qe2 Kf8 Qd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6 Qd2 cxd5 cxd5 a6 Bd3 Nh5 g4 Nf4 Bc2 b5 Qf2 Nd7 Nge2 b4 Na4 a5 Nxf4 exf4 Bxf4 Ne5 O-O-O Nc4 Be3 Nxe3 Qxe3 Rb8 Bb3 Bd7 Kb1 Qe8 Nb6 Bb5 Rd2 a4 Bd1 Rb7 e5 b3 axb3 axb3 Bxb3 Qb8 Nc4 Bxc4 Bxc4 Bxe5 Qe2 Qa7 Rc1 Ra8 b3 Bf4 Kc2 Re7 Qd3 Qc5 Rb1 Re3 Qd4 Ra2 Kd1 Rxf3 Qxf4 Rxf4 Rxa2 Qg1 Kc2 Qxh2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 a4 Nc6 Be3 Be7 O-O O-O f4 Qc7 Kh1 Re8 Bd3 Nb4 a5 Bd7 Qf3 Rac8 Qg3 Nh5 Qf3 Nf6 Qg3 Nh5 Qf3 Nf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bb5 c6 Ba4 O-O Ne2 b5 Bb3 a5 O-O Nd7 Bg5 Nb6 a3 Qc7 Qd2 c5 Bh6 Bxh6 Qxh6 Ba6 Nf4 c4 Bc2 e5 Ne2 b4 axb4 axb4 f4 exd4 f5 Nd7 Nxd4 Qe5 Nf3 Qg7 Qxg7 Kxg7 cxb4 Bb7 Kf2 Nf6 Rxa8 Rxa8 Ke3 Re8 fxg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 e6 O-O Nge7 c3 d5 exd5 Qxd5 Re1 g6 b4 Bg7 Bb2 O-O c4 Qh5 Bxg7 Kxg7 bxc5 Qxc5 d4 Qb6 Nc3 Rd8 Bxc6 Qxc6 Qe2 a6 Rac1 Qc7 d5 f6 dxe6 Rd6 Nd5 Qd8 Nxf6 Rxe6 Ne4 Nc6 Qb2 Kg8 Nfg5 Re5 f4 Rf5 Rcd1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 c5 Bxb4 cxb4 Qb3 a5 a3 Na6 Nbd2 d6 g3 b6 Bg2 Bb7 O-O O-O Rfd1 Qe7 Nf1 Be4 N1d2 Bb7 Nh4 Bxg2 Nxg2 Qb7 Qd3 Rfe8 e4 e5 f3 bxa3 bxa3 Nc7 Ne3 Ne6 Nf5 Qc7 d5 Nc5 Qe3 Qd8 Rab1 g6 Nh6 Kg7 Nf1 Nfd7 h4 Rf8 Rb2 f5 exf5 gxf5 Qg5 Qxg5 hxg5 Kg6 f4 Rae8 Rf2 exf4 gxf4 Re4 Nd2 Rd4 Re1 Ne4 Nxe4 Rxe4 Rxe4 fxe4 Ng4 Rc8 Ne3 b5 Rh2 h5 gxh6 Kh7 cxb5 Rc3 Nf5 Rc1 Kf2 Nb6 Nxd6 Rc2 Kg3 Rc3 Kg4 Nxd5 Nxe4 Rc4 Kf3 Ra4 Ng5 Kh8 Rd2 Rxf4 Kg3 Rf5 Kg4 Rf4 Kh5 Rf5 b6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 Nbd2 a5 a4 Na6 Nb3 b6 c3 Bb7 Bf4 c5 Ne5 e6 Qc1 Rc8 Bh6 cxd4 Nxd4 Nd7 Bxg7 Kxg7 f4 Nac5 Qe3 Re8 g4 f6 Nxd7 Qxd7 g5 Qe7 gxf6 Qxf6 Nf3 Qe7 Rad1 Kg8 Bh3 Nxa4 Nd4 Qg7 Nxe6 Nc5 f5 gxf5 Nxg7 Rxe3 Nxf5 Rxh3 Ne7 Kg7 Nxc8 Bxc8 Rxd5 Rh6 Rf4 Re6 Rg5 Rg6 Rxg6 hxg6 Kf2 b5 Ke3 Na4 c4 b4 Kd4 Nxb2 c5 a4 c6 a3 Rf1 a2 c7 b3 Kc3 Nc4
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O d3 d6 c3 Bb7 Nbd2 Na5 Bc2 c5 Nf1 Re8 Bg5 h6 Bh4 Nh5 Bxe7 Qxe7 Ne3 Nf4 g3 Ng6 a3 Nc6 b4 Rad8 Rb1 Qd7 h4 Qh3 Qe2 Nce7 a4 Bc6 axb5 axb5 Qf1 Qe6 Nd2 cxb4 Rxb4 d5 Bb3 Qd7 Nf3 Qd6 Rb1 dxe4 dxe4 Qc5 Rc1 Rd6 Qe1 Rc8 h5 Nf8 c4 Be8 Rc2 Rd3 cxb5 Qd6 Rxc8 Nxc8 Bc2 Rxe3 fxe3 Nb6 Rb1 Bd7 Qc3 Bg4 Nxe5 Nfd7 Nxg4 Qxg3 Kf1 Qxg4 Ke1 Qg3 Kd2 Ne5 Rd1 Nec4 Kc1 Qb8 e5 Nxe5 Qxe5 Qa8 Kb1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 Ne8 Rb1 f5 b4 Nf6 f3 f4 c5 g5 Nc4 Ng6 a4 Rf7 Ba3 Bf8 a5 Rg7 b5 b6 cxd6 cxd6 Na2 g4 fxg4 Nh4 Rb3 Nxg4 Bf3 bxa5 Qd2 a4 Rc3 Bd7 Rb1 Rc8 Kh1 Nxh2 Kxh2 Rxc4 Rbc1 Nxf3 Rxf3 Qh4 Kg1 Rxe4 Qf2 Qh5 Rc7 Re1 Qxe1 Qxf3 Qd2 Qg4 Nc3 f3 Rxa7 Qh3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Qb6 a3 Nbd7 Nb3 Qc7 g4 h6 h4 h5 g5 Ng4 Bd2 b5 Bh3 Nb6 O-O-O Nc4 Bxg4 hxg4 Qxg4 Rb8 g6 a5 Nd4 b4 Ncb5 Qc5 a4 e5 Nf5 f6 Qe2 Bxf5 exf5 Rxb5 axb5 a4 Kb1 Qxb5 Qd3 b3 Ka1 Be7 Bc1 Kd7 Rh2 Rc8 h5 e4 Qd5 Rc5 Qe6 Kd8 c3 a3 bxa3 Qa4 Rd4 Qa6 Rxe4 Rc7 Qg8 Kd7 Rhe2
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 c4 c6 d4 e6 Qc2 Nf6 Bg5 h6 Bh4 Nbd7 e3 Be7 Nc3 O-O Bd3 dxc4 Bxc4 Nd5 Bg3 Nxc3 bxc3 c5 O-O Nb6 Bd3 Bd7 Bh7 Kh8 Be4 Qc8 Ne5 Ba4 Qe2 Nd7 Rab1 Bc6 Nxc6 bxc6 Qf3 Nb8 Be5 f5 Qg3 Bf6 Bd3 Bxe5 Qxe5 cxd4 cxd4 Rf6 Rfc1 Nd7 Qa5 f4 Be4 fxe3 fxe3 Rb8 Qxa7 Rxb1 Rxb1 Rf8 Qb7 Qd8 Bxc6 Nf6 Bf3 Nd5 Bxd5 exd5 Rf1 Re8 Qb3 Qa5 h3 Ra8 Rf5 Rd8 Kh2 Qc7 Re5 Qf7 a4 Kh7 Qc2 Kh8 a5 Qa7 Qc5 Qa8 Qc7 Rc8 Qd7 Rd8 Qb5 Kh7 a6 Rd6 Qb7 Qxa6 Qxa6 Rxa6 Rxd5 Kg6 Kg3 Kf6 Kf3 Ra3 h4 Rb3 g4 Ke6 Ra5 Kf6 Kf4 Rb6 e4 Rb1 Ra6 Kf7 g5 h5 g6 Ke7 Ke5 Rb7 d5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d6 d4 Bg4 e4 Nf6 h3 Bh5 Bd3 e6 c4 Be7 Nc3 Nc6 Be3 O-O Qe2 Bg6 Rd1 d5 cxd5 exd5 e5 Ne4 O-O Qd7 Bc1 f5 Nxd5 Bh5 Nf4 Bxf3 Bc4 Kh8 Qxf3 Ng5 Qh5 Qe8 Qe2 Bb4 Nd5 Ne6 Nxb4 Nxb4 d5 Nc5 a3 Nba6 b4 Ne4 Bb2 Nb8 e6 Qe7 Bd3 Nd6 Qe5 b5 Rfe1 a5 Re3 axb4 axb4 Na6 Bc3 Rf6 Qf4 Rg6 Ra1 Qf8 Rg3 Kg8 h4 Ne8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 a4 g6 Bd3 Bg7 Nf3 Nc6 O-O Bg4 Qe1 O-O Qh4 Bxf3 Rxf3 e6 Be3 Qd8 Raf1 d5 f5 dxe4 Rh3 exf5 Bxe4 Re8 Bxf5 gxf5 Rxf5 Rxe3 Rxe3 Qb6 Qf2 Ng4 Re8 Rxe8 Qxb6 Bd4 Qxd4 Re1 Rf1 Rxf1 Kxf1 Nxd4 Ne4 Nxc2 Nc5 b5 a5 Nb4 h3 Ne3 Ke2 Nc4 b3 Nxa5
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 e5 Bg2 d6 d3 f5 e3 a5 Nge2 Nf6 O-O O-O b3 c6 Bb2 Na6 Qd2 Bd7 Kh1 Rc8 Rae1 b5 e4 Nc5 cxb5 cxb5 exf5 Bxf5 Ne4 b4 Rc1 Nfxe4 dxe4 Bd7 Rcd1 Bb5 Qe3 a4 Rfe1 Qb6 Nc1 a3 Ba1 Bd7 Nd3 Be6 f4 Rc6 Nf2 Nd7 Qd2 Nc5 Bh3 Bf7 fxe5 dxe5 Ng4 Be6 Nh6 Kh8 Bxe6 Nxe6 Ng4 Nd4 Bxd4 exd4 Nf2 Rc3 Nh3 d3 Nf4 Rc2 Qxd3 Rxa2 Rf1 Rf2 Rxf2 Qxf2 Rd2 Qa7 Qe2 Bc3 Rc2 Qf7 e5 Qb7 Qg2 Qxg2 Kxg2 Bb2 Rf2 a2 Nxg6 hxg6 Rxf8 Kg7 Rf1 Bxe5 Kf3 a1=Q Rxa1 Bxa1 Ke4 Kf6
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 c5 exd5 Qxd5 dxc5 Bxc5 Ngf3 Nf6 Bd3 O-O Qe2 Nbd7 Ne4 b6 Nxc5 Qxc5 Be3 Qc7 Bd4 Bb7 O-O-O Nc5 Be5 Nxd3 Rxd3 Qc4 Nd4 Be4 Re3 Qxa2 Bxf6 Bg6 Ra3 Qd5 h4 gxf6 h5 Qxd4 hxg6 hxg6 Rah3 f5 Rh4 f4 Qf3 Rac8 Rxf4 Qc5 c3 Kg7 Rhh4 Qe5 g3 Qe1 Kc2 Rcd8 Rd4 Qe5 Rhf4 Qc7 Qe3 e5 Rxd8 Rxd8 Re4 Rd5 g4 b5 g5 Qd6 f3 a5 Qe2 Qe6 Qh2 Qf5 Qg3 Qd7 Qe1 b4 cxb4 Qa4 b3 Qa2 Kc3 a4 bxa4 Qa3 Kc2 Qxa4 Kc3 Qa3 Kc2 Rd3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6 Bd3 cxd5 cxd5 Nh5 Nge2 f5 exf5 gxf5 O-O Nd7 Rc1 Nc5 Bc4 a6 b4 Nd7 a4 Qh4 f4 Kh8 Qd2 Rg8 g3 Bh6 Rf2 Qh3 Rg2 Nhf6 Kh1 Ng4 Ng1 Qh5 Be2 Ndf6 Bb6 Bd7 h3 Rae8 Rf1 Rg7 b5 axb5 axb5 Rge7 Qb2 exf4 gxf4 Qh4 Nd1 Bg7 Bd4 Ne4 Rf3 Rg8 Bf1 Bxd4 Qxd4 Reg7 Nc3 Qf6 Nge2 Qxd4 Nxd4 Ngf6 Rxg7 Rxg7 b6 Nxc3 Rxc3 Nxd5 Rb3 Nxf4 Kh2 Rg6 Bb5 Bxb5 Rxb5 Rg2 Kh1 Rd2 Nxf5 Kg8 Rb4 Nxh3 Rc4 Nf2 Kg1 Nh3 Kh1 Kf7 Rc7 Ke6 Ne3 Ng5 Rxb7 h5 Rg7 Nf3 Rg2 Kd7 Rxd2 Nxd2 Kg2 Kc6 Nf5 Ne4 Kh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 g6 Nc3 Bg7 e4 O-O d4 d6 Be2 e5 Be3 c6 Qd2 Nbd7 Rd1 Re8 d5 cxd5 cxd5 a6 O-O b5 Qc2 Nb6 a4 bxa4 Nxa4 Nxa4 Qxa4 Rb8 Nd2 Ng4 Ba7 Rxb2 Bxa6 Re7 Bxc8 Qxc8 h3 Qa8 Ra1 Nf6 Nc4 Rbb7 Be3 Qb8 Na5 Rb4 Qd1 Rxe4 Nc6 Qf8 Nxe7 Qxe7 Ra8 Bf8 Qf3 Rxe3 fxe3 Nd7 Rb1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Nd3 f5 Bd2 Nf6 f3 f4 g4 g5 b4 h5 h3 Kf7 Be1 Rh8 Kg2 Ng6 c5 hxg4 hxg4 Nh5 Rh1 Ng3 Bxg3 fxg3 Qd2 Rh4 Kxg3 Bd7 Rxh4 gxh4 Kh2 Bf6 b5 Bg5 Qe1 Be3 c6 Bc8 Nd1 Bd4 cxb7 Bxb7 Rc1 a6 Qd2 axb5 Qh6 Nf8 Rc2 Bb6 Ne3 Qf6 Qh5 Qg6 Nf5 Qxh5 gxh5 Bc8 Nxh4 Nh7 f4 exf4 Nxf4 Ng5 Bxb5 Nxe4 a4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 g3 g6 Bg2 Bg7 a3 Nc6 Rb1 a5 d3 O-O Bg5 d6 Nf3 h6 Bd2 Be6 h3 d5 cxd5 Nxd5 O-O Nd4 e4 Nxf3 Qxf3 Nc7 e5 Bf5 Rbd1 Ne6 Qxb7 Rb8 Qa7 Bxd3 Bc1 c4 Rfe1 Nd4 Nd5 Nb3 Nxe7 Kh7 Be3 Bxe5 Nc6 Qf6 Nxe5 Qxe5 Bf4 Qd4 Bxb8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5 e4 c6 b3 exd4 Nxd4 Re8 h3 Nc5 Re1 Bd7 Bf4 Qb6 Be3 Rad8 Qc2 Bc8 a3 Qc7 Rad1 Qe7 b4 Ncd7 Qa2 a6 a4 a5 b5 c5 Nde2 Nb6 Nf4 Be6 Nxe6 Qxe6 Nd5 Nfxd5 exd5 Qf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 Nbd2 a5 c4 c6 b3 Be6 Bb2 a4 bxa4 Qa5 cxd5 cxd5 Ne5 Nbd7 Nxd7 Bxd7 Nb3 Qxa4 Nc5 Qxd1 Rfxd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 c4 e6 d4 Nf6 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Qc2 Nc6 a3 Qa5 O-O-O Be7 g4 dxc4 Bxc4 e5 g5 exf4 gxf6 Bxf6 Nd5 Ne7 Nxf6 gxf6 Rhg1 Kh8 Qe4 Ng6 Qd4 Qb6 Qxb6 axb6 Rd6 fxe3 fxe3 Ra5 Kd2 Rf5 Rf1 Rc5 Bd3 Rc6 Rxc6 bxc6 b4 Rd8 Rc1 Ne5 Nd4 Nxd3 Kxd3 Bd7 Ke4 Re8 Kf4 Rc8 Nf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Bd3 O-O Nge2 c5 d5 e6 O-O exd5 cxd5 Re8 h3 Nbd7 Ng3 a6 a4 Rb8 a5 h5 Bg5 b5 axb6 Qxb6 Ra2 Ne5 Be2 Nh7 Be3 h4 Nh1 g5 Qd2 Qd8 Qc2 Nf8 Qc1 Bf6 f3 Nfg6 Nf2 Rb6 Rd1 Nf4 Bf1 Nh5 Qc2 Ng3 b3 Re7 Nb1 Ng6 Nd2 Nf4 Nc4 Rb4 Nd3 Ng6 Ra3 Rc7 Qd2 Rb8 e5 dxe5 d6 Rc6 Nxc5 Bf5 Qd5 Rxc5 Qxc5 e4 fxe4 Bxe4 Qc7 Nf4 Qxd8 Bxd8 Bxf4 gxf4 Bd3 Bb7 Re1 Bf6 d7 Bc6 Re8 Rxe8 dxe8=Q Bxe8 Rxa6 Bd4 Kh2 Bb5 Rd6 Bf2 Rd8 Kg7 b4 f5 Rb8 Ba4 Nd2 Kf6 Ra8 Bd7 b5 Ke7 Ra6 Be6 Ra4 Bd5 Ra1 Be4 Bf1 Kd6 Nf3 Kc5 Rc1 Kb6 Bc4 Kc5 Bf1 Kb6 Ne5 Bd4 Nd7 Ka5 Rd1 Nxf1 Rxf1 Kxb5 Nf8 Kc4 Ng6 Bf6 Rxf4 Bg5 Rf1 Kd3 g3 Ke2 Ra1 f4 Ra2 Ke3 Ra3 Kf2 gxf4 Bh6 Ra2 Kf1 Ra1 Kf2 Rg1 Bxg6 Rxg6 Bxf4 Kh1 Bg3 Rg4 Kf1 Rg5 Kf2 Rf5 Ke3 Kg2 Ke4 Rf7 Kd5 Re7 Kd6 Re2 Kd5 Kf3 Kd6 Kg4 Kd7 Kf5 Kd6 Re8 Kd7 Re6 Kc7 Ke4 Kd7 Kd5 Kc7 Re7 Kb6 Rf7 Bh2 Rf2 Bg3 Rc2 Kb7 Rc6 Bf4 Kc5 Bg3 Kd5 Bf4 Rc2 Bg3 Ke6 Kb6 Kd7 Kb5 Rc6 Bh2 Rc2 Bg3 Ke6 Kb6 Kf5 Kb5 Ke4 Kb6 Kf3 Kb5 Kg4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 e5 g3 c6 d4 exd4 Qxd4 d5 Bg2 Be6 cxd5 cxd5 Nf3 Nc6 Qa4 Bc5 O-O O-O Ne1 d4 Nd3 Bb6 Nb5 Bd5 Bh3 Re8 Nf4 Be4 Rd1 Qe7 Na3 d3 exd3 Bf3 Rf1 Ng4 Bd2 Nce5 Bg2 Nxf2 Rxf2 Bxg2 Kxg2 Bxf2 Kxf2 g5 Qe4 gxf4 gxf4 Qh4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 Nf3 e6 Bd3 b5 Qe2 Nbd7 O-O Bb7 Kh1 Be7 e5 dxe5 fxe5 Nd5 Ne4 O-O Neg5 h6 Nh7 Rfe8 c3 Rad8 Bb1 Qc4 Bd3 Qg4 Bd2 Nc5 Bc2 Qc4 Qf2 Ne4 Qe1 Nb4 Nf6 Nxf6 Bb3 Nd3 exf6 Nxe1 Bxc4 Nxf3 fxe7 Rxd2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 Bxc6 bxc6 O-O Bg7 Re1 f6 d4 cxd4 Nxd4 e5 Nb3 Nh6 c4 Nf7 c5 O-O Na3 a5 Nc4 a4 Nb6 Ra7 Nxc8 Qxc8 Nd2 Qa6 Qc2 Ng5 Nc4 Ne6 Be3 Rb8 Red1 Bf8 Nb6 Rbb7 Nc8 Ra8 Nb6 Raa7 b4 axb3 Qxb3 Qb5 Qxb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 g3 d5 Bg2 Nf6 d4 Be7 Nf3 O-O Qc2 dxc4 Qxc4 a6 Bf4 Nd5 Nc3 b5 Qd3 Bb7 Nxd5 Bxd5 Rc1 Bb4 Kf1 Ra7 Ng5 f5 e4 Be7 exd5 Bxg5 dxe6 Bxf4 gxf4 c6 d5 Re7 Rd1 cxd5 Bxd5 Qb6 Qe3 Qd6 Kg2 a5 Rhe1 Kh8 Bf3 Qb4 Qe5 h6 Qd6 Qxd6 Rxd6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 e4 Nf6 f3 g6 Be3 Bg7 Qd2 e5 dxe5 dxe5 Qxd8 Kxd8 Bc4 Be6 Bxe6 fxe6 Nd2 Nc6 c3 b6 Ke2 Ke7 a4 Rad8 Nh3 Rd7 Nf2 Rhd8 Nc4 Ne8 b4 Kf7 Rad1 Nd6 Na3 a6 Nd3 a5 b5 Ne7 c4 Nec8 c5 bxc5 Nxc5 Re7 Nc4 Nxc4 Rxd8 Nxe3 Kxe3 Nd6 Ke2 Bh6 Rd1 Nc4 R1d7 Bg5 Rc8 Be3 Rcxc7 Bxc5 Rxe7 Bxe7 Rxc4 Bd8 Rc8 Ke7 Kd3 Kd7 Rxd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 e4 c6 b3 Qc7 h3 Re8 Be3 exd4 Nxd4 Nc5 Qc2 a5 Rfe1 Bd7 Rad1 Rad8 f4 Re7 Bf2 Rde8 a3 Na6 Nf3 Be6 Qd2 Rd7 Nd4 h6 f5 Kh7 fxe6 fxe6 Qc2 e5 Nf3 Rf8 Rb1 Rdf7 Be3 Nh5 Qf2 Nc5 Bxc5 dxc5 Ne2 b6 Qe3 Nf6 Rbd1 Rb8 Nc3 Nh5 Nh2 Rd7 Rxd7 Qxd7 Rd1 Qe6 Ng4 g5 Qe1 b5 cxb5 cxb5 Nd5 b4 axb4 axb4 Nge3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 Bb4 Bd2 a5 Nc3 b6 e3 Bb7 Bd3 O-O Qc2 d6 O-O Nbd7 e4 e5 Nd5 Bxd2 Qxd2 c5 dxe5 Nxe5 Nxe5 dxe5 Rfe1 Qd6 Bc2 Bxd5 exd5 Rae8 Re2 e4 h3 Re5 Rae1 Rfe8 Qc3 R5e7 Ba4 Rd8 Re3 h6 a3 Qf4 Qc2 Rb7 Qe2 Re7 Qc2 Rb7 Bc6 Ra7 Qc3 Re7 Qb3 Rb8 Qc2 Rc7 Qc3 Re7 b4 axb4 axb4 Ra7 Qb2 Qd6 Rb1 Rf8 bxc5 bxc5 Ra3 Re7 Qe2 Nd7 Bxd7 Qxd7 Qe3 Qd6 Rab3 Rc8 Rb7 Rxb7 Rxb7 f5 g3 Qa6 Qb3 Qa1 Kg2 Qd4 Rb8 Rf8 Rxf8 Kxf8 Qb8 Kf7 Qf4 g6 Qc7 Kf6 Qd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Nc3 Nf6 e5 dxe5 Nxe5 a6 a4 Qc7 Nc4 Nc6 b3 b6 Bd3 Bb7 O-O Nb4 Be2 g6 Bb2 Bg7 Bf3 O-O Bxb7 Qxb7 d3 Rfd8 Qe2 Nc6 Rfe1 Nd4 Qd1 e6 Qc1 b5 Ne3 Qc6 h3 Nd7 Ne2 h5 Nf1 e5 Nd2 Nb6 Nc3 Re8 Qd1 Rad8 axb5 axb5 Ra5 c4 bxc4 bxc4 Nce4 cxd3 cxd3 Nf5 Qb3 Nd5 Nc5 Qc8 Nb7 Rd7 Rxd5 Rxb7 Rb5 Rxb5 Qxb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d6 Nf3 Bg4 e4 Nd7 h3 Bh5 Nc3 e6 Bd3 Be7 Qe2 c6 g4 Bg6 Bf4 h5 O-O-O Qa5 Kb1 O-O-O g5 h4 Nd2 e5 Be3 Kb8 d5 Bf8 Nc4 Qc7 Na3 Nb6 dxc6 bxc6 Ba6 f5 exf5 Bxf5 f4 e4 Nab5 cxb5 Nxb5 Qc4 Qf2 Bc8 Rd4 Qc6 Nxa7 Kxa7 Rc4 Qd5 Bxb6 Kxa6 Ra4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Nc3 d5 Bf4 Bg7 e3 Bg4 Be2 c6 Ne5 Bxe2 Qxe2 Nh5 h4 Nxf4 exf4 h5 O-O-O Nd7 Kb1 e6 g3 Bxe5 fxe5 Qa5 Rhe1 b5 Qe3 Qb6 Ne2 a5 Nf4 c5 dxc5 Nxc5 Nd3 Rc8 Qf4 Nd7 f3 Rc4 Qg5 b4 Nf4 Rc6 Re2 Qd8 Nh3 Qxg5 Nxg5 Ke7 Rd4 Rhc8 Rf4 Nxe5 Rxe5 f6 Rxe6 Rxe6 Nxe6 Kxe6 Rd4 g5 Kc1 Ke5 Rd1 gxh4 gxh4 Rc4 Rh1 Kf4 Kd2 f5 Rh3 Rc7 c3 bxc3 bxc3 Rc4 Kc2 d4 Kd3 Rxc3 Kxd4 Rxf3
Result: 0-1
Processing game...
FEN: 
Moves: c4 d6 g3 Nf6 Bg2 g6 Nc3 Bg7 d3 O-O Nf3 e5 O-O h6 Rb1 Nbd7 b4 Re8 Nd2 c6 b5 c5 Bb2 Rb8 e3 Nb6 a4 Be6 a5 Nc8 Ra1 Nh5 a6 b6 Qb3 f5 Rae1 Bf7 f4 g5 Nd5 Qd7 fxg5 hxg5 Qd1 g4 Rf2 Be6 Ref1 Rf8 Nb1 Ne7 Nxe7 Qxe7 Nc3 Rf7 Bd5 Rbf8 Bxe6 Qxe6 Nd5 Bh6 Qe2 Bg5 Bc1 Qh6 Bd2 Ng7 e4 Bxd2 Qxd2 f4 gxf4 exf4 Rxf4 Kh8 Qe3 Rxf4 Rxf4 Rxf4 Qxf4 Qh4 Kf1 Ne6 Qf6 Qxf6 Nxf6 Kg7 Nxg4 Nf4 Ne3 Ne6 Nf5 Kf6 Nxd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d3 Nc6 g3 g6 Bg2 Bg7 O-O e5 c4 Nge7 Ne1 O-O Nc3 f5 Nc2 f4 a3 g5 g4 h5 f3 hxg4 fxg4 Ng6 Bd2 Be6 Be1 Qd7 Bf3 Nd4 h3 a6 Nd5 Rab8 a4 b5 axb5 axb5 Na3 Bxd5 exd5 b4 Nb5 Nxb5 Be4 Ne7 cxb5 Qxb5 Ra4 Bf6 Rf2 Kg7 Rh2 Rh8 b3 Qd7 Raa2 Ng6 Qa1 Nh4 Ra7 Rb7 Rxb7 Qxb7 Ra2 Qf7 Ra7 Be7 Bf2 Ng6 Bxc5 dxc5 Bxg6 Kxg6 Qxe5 Qf6 Rxe7 Qxe5 Rxe5 Rxh3 d4 cxd4 d6 Rh8 Rd5 Kf6 Rxd4 Ke6 d7 Rd8 Rxb4 Rxd7 Rb6 Rd6 Rb5 Rd5 Rb6 Ke5 Kf2 Rd2 Kf3 Rd3 Kf2 Rg3 Rb5 Kf6 Rb6 Ke7 Rb7 Kd6 Rg7 Rxg4 Kf3 Rg3 Ke4 f3 Rf7 g4 b4 Kc6 Rf5 Rg2 Kf4 f2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 Nd2 a5 Rb1 Nd7 a3 f5 b4 Kh8 f3 f4 Nb3 axb4 axb4 g5 Bd2 Ng6 Ra1 Rxa1 Qxa1 Nf6 Qa7 g4 fxg4 Nxg4 h3 Nh6 Be1 Rg8 Nd2 Bf6 Kh1 Bh4 Nf3 Bxe1 Nxe1 Nh4 Rf2 Qg5 Nf3 Nxf3 Bxf3 Bxh3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 f4 a6 Be2 Be7 O-O Qc7 Kh1 O-O a4 Nc6 Be3 Re8 Bf3 Rb8 Qd2 Bd7 Nb3 b6 Qf2 Nb4 Nd4 e5 Nf5 Bxf5 exf5 e4 Nxe4 Qxc2 Rfc1 Qxf2 Nxf2 d5 Bd2 a5 g3 Bc5 Bxb4 Bxb4 Kg2 h5 Rc2 b5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 e3 Bf5 Nge2 O-O Ng3 Be6 Bd3 c5 dxc5 Bxc5 O-O Nc6 Rc1 Bd6 Nh5 Be7 Nb5 Nxh5 Qxh5 g6 Qf3 Rc8 Rfd1 Qd7 h3 Rfd8 Qg3 Nb4 Nc3 Nxd3 Rxd3 Bf5 Rd2 Qe6 Rcd1 h5 h4 Rc5 f3 Qc6 e4 Rxc3 bxc3 Qb6 Kh2 dxe4 Rxd8 Bxd8 Be3 Qa5 Qb8 Qc7 Qxc7 Bxc7 Kg1 exf3 gxf3 b6 Kf2 Kf8 Rd4 Ke7 Bf4 Bxf4 Rxf4 Kd6 Ke3 Kc5 Rd4 Be6 a3 a5 Ke4 b5 Ke5 a4 f4 Kc6 Kf6 Kc5 Rb4 Bc4 Ke7 Be6 Re4 Kd5 Rd4 Kc6 Kd8 Bf5 Ke8 Be6 Kf8 Kc5 Kg7 Kc6 Kg8 Kc5 Kf8 Kc6 Kg7 Kc5 Kh8 Kc6 Kh7 Kc5 Kh6 Bf5 Kg5 Be6 Kf6 Kc6 f5 Bxf5 Kxf7 Kc5 Kf6 Bc2 Ke7 Bf5 Kd8 Kc6 Rf4 Kd6 Rb4 Kc5 Kc7 Bd3 Rd4 Be2 Kb7 Bf1 Ka7 Be2 Re4 Bd3 Rb4 Bc4 Ka6 Kd5 Ka5 Ke5 Rb1 g5 hxg5 Kf5 Rg1 Kg6 Kb4 h4 Kc5 h3 Kd4 Be6 Ke5 Bd7 Kf4 Bc6 Kg3 Kxg5 Rd1 h2 Kxh2 Kf4 c4 bxc4 Rd4 Ke5 Rxc4 Kd5 Rb4 Kc5 Kg3 Bb5 Kf4 Kb6 Ke3 Ka5 Kd4 Be2 Rb1 Bh5 Re1 Bf7 Kc5 Bb3 Re8 Ka6 Kc6 Ka7 Kb5 Kb7 Re7 Kc8 Kc6 Kd8 Rd7 Ke8 Kc7 Bc2 Rd2 Bb3 Re2 Kf7 Kd6 Bc4 Re7 Kf8 Re4 Bb3 Kd7 Kf7 Rf4 Kg6 Kd6 Kg5 Ke5 Kg6 Rf3 Kg7 Rf6 Bc4 Kf5 Bb3 Kg5 Bc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 O-O-O a6 Kb1 b5 Nc1 exd4 Bxd4 b4 Na4 c5 Bxf6 Bxf6 Qxd6 Be7 Qg3 Bh4 Qh3 Be7 Qg3 Bh4 Qh3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 a5 Nc3 Bb4 a3 Bxc3 bxc3 Nbd7 exd5 Nxd5 O-O O-O Re1 Re8 c4 Ne7 Ng5 h6 Ne4 a4 Ba2 c5 Nd6 Rf8 c3 Ng6 Bb1 Nf6 Nxc8 Qxc8 Qf3 Re8 Bc2 Nh4 Qg3 g5 d4 Nf5 Qh3 e4 f3 exf3 Rxe8 Qxe8 Bxg5 hxg5 Qxf5 Qe3 Kh1 fxg2 Kxg2 Qe2 Kg1 Ng4 Qh7 Kf8 Rf1 f6 Bg6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 e6 d4 d5 exd5 exd5 Nf3 Nc6 Be2 Bd6 dxc5 Bxc5 O-O Nge7 Nbd2 O-O Nb3 Bb6 Bg5 h6 Bxe7 Qxe7 Re1 Be6 a4 a5 Qd3 Rad8 Qb5 Qc7 Nbd4 Bg4 Bf1 g6 g3 h5 Qd3 Rfe8 Bg2 Kg7 Kf1 Qd6 Qb5 Nxd4 Nxd4 Rxe1 Rxe1 h4 Qd3 Qf6 Kg1 h3 Bf1 Rc8 Qd2 Bd7 Nb5 Rd8 Nd4 Rc8 Nb5 Rh8 Re2 Bg4 Re1 Bd7 Re2 d4 Nxd4 Bxa4 Re4 Bd7 Qe3 Bc6 Rf4 Bxd4 Qxd4 Qxd4 Rxd4 Re8 f4 a4 Rd2 Re1 Kf2 Rc1 Bxh3 a3 bxa3 Rxc3 Bg2 Rxa3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 d5 e3 Nbd7 cxd5 exd5 Bd3 Re8 Ne2 h6 Bh4 c5 O-O Rc8 Qd2 Qe7 Bf2 Bc6 Nc3 Nf8 Rfe1 Ne6 Bh4 Ng5 Bf5 Bd7 Bc2 Bc6 Rad1 Qe6 Qf2 Rcd8 h3 Ngh7 dxc5 bxc5 e4 dxe4 Rxd8 Rxd8 Qxc5 Ng5 Bxg5 hxg5 Qxg5 Qc4 fxe4 Qd4 Qe3 Qxe3 Rxe3 Rd2 Re2 Rxe2 Nxe2 Nxe4 Bxe4 Bxe4 Kf2 Kf8 g3 Ke7 Ke3 Bc6 h4 Bd7 Kf4 Kd6 g4 f6 h5 Ke7 Nd4 Kf7 b4 Ba4 Nf5 g6 Nd6 Kg7 Nc8 a6 Nd6 Bd1 Ne8 Kf7 Nxf6 Kxf6 g5 Kf7 h6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 O-O-O a6 Bh6 Bxh6 Qxh6 b5 h4 Qa5 h5 b4 Nb1 Qxa2 Ng3 Nb6 c5 Nc4 Rd2 Na5 cxd6 Nb3 Kc2 Na1 Kc1 Nb3 Kc2 Na1 Kc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Nh4 Bg6 h3 O-O Nxg6 hxg6 Qc2 Rc8 Rd1 Qb6 e4 c5 d5 Ne5 Be2 exd5 Nxd5 Nxd5 Rxd5 Nc6 Bc4 Nd4 Qd3 Rcd8 Be3 Rxd5 Bxd5 Rd8 Qc4 Rd7 Rc1 Qf6 Rd1 Ne6 Qb3 a5 Rd3 Nf4 e5 Qf5 Bxf4 Qxf4 e6 Rd8 e7 Re8 Rf3 Qc1 Kh2 Rxe7 Bxf7 Kh7 Bxg6 Kh6 Qd5 Qg5 Bf5 g6 h4 Qf6 Bd3 Qe5 Qxe5 Rxe5 Rf6 c4 Bxc4 Be7 Rb6 Bc5 Rf6 Re4 Bd3 Rg4 Kh3 Be7 Re6 Rxh4 Kg3 Rd4 Rxg6 Kh5 Bf5 Bd6 Kf3 Bc5 g4 Kh4 Rh6 Kg5 Rg6 Kh4 Be4 Rd6 Rg7 Rf6 Bf5 Rb6 Rh7 Kg5 Rh5 Kf6 Bd3 Bd4 g5 Kg7 Rh7 Kf8 Bc4 Rxb2 Rf7 Ke8 g6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 Rd1 a6 dxe5 Nxe5 b3 b5 cxb5 axb5 Qxd6 Nfd7 f4 b4 Nb1 Ng4 Bd4 Bxd4 Qxd4 Rxa2 h3 c5 Qg1 Ngf6 e5 Ne4 h4 c4 Nc1 c3 Nxa2 c2 Qd4 cxd1=Q Kxd1 Ndc5 Qxd8 Rxd8 Kc2 Nf2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rad1 Bd7 Nf3 b4 Ne2 a5 Nf4 Kh8 Bg5 Nf6 Qh4 Bb5 Nd4 Be8 Ndxe6 fxe6 Nxe6 Qa7 e5 dxe5 Nxf8 Bxf8 Bxf6 gxf6 Rd8 Nd7 Qg4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 h3 c6 e4 Qb6 c5 dxc5 dxe5 Ne8 e6 fxe6 Ng5 Ne5 f4 Nf7 Nxf7 Bd4 Kh2 Rxf7 e5 Nc7 Ne4 Nd5 a4 a5 Ra3 Qc7 Nd6 Rf8 h4 b6 h5 Qg7 hxg6 hxg6 Be4 Ba6 Re1 Ne7 Kg2 Rad8 Rb3 Bxe5 fxe5 Qxe5 Qg4 Rxd6 Bf4 Rxf4 gxf4 Rd2 Kg3 Qf6 Rxb6 Bd3 Rb8 Kf7 Qh3 Bxe4 Qh7 Qg7 Qxg7 Kxg7 Rxe4 Nf5 Kf3 Kf6 Rc4 Rd5 Rf8 Ke7 Ra8 Nd6 Rc3 c4 Ra6 Rc5 b3 Kf6 bxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Be3 a6 f3 Nbd7 g4 h6 Rg1 Qb6 a3 Ne5 Bf2 Qc7 f4 Nc4 Bxc4 Qxc4 Qf3 e5 Nf5 Bxf5 gxf5 d5 fxe5 Nxe4 Rg4 h5 Rh4 Bc5 O-O-O Bxf2 Nxe4 dxe4 Qxf2 Rc8 Kb1 Rd8 Rxd8 Kxd8 Rh3 Qd5 Rc3 Kd7 Qb6 Rd8 Rc5 Qd1 Ka2 Ke8 Qxb7 Qg4 e6 fxe6 Re5 Qg5 h4 Qxh4 Rxe6 Kf8 f6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Nf3 Ngf6 Ng3 e6 Bd3 Be7 O-O O-O Qe2 b6 c4 Bb7 Bf4 c5 dxc5 Bxc5 Rad1 Qe7 Ng5 h6 N5e4 Nxe4 Nxe4 Rad8 Bc2 Nf6 Nxc5 Qxc5 b3 a5 Qe3 Qc6 f3 b5 Be5 Nd7 Bd6 Rfe8 Be4 Qa6 Bc7 Bxe4 fxe4 bxc4 Bxd8 Rxd8 Qf2 f6 e5 f5 Rd6 Qc8 Qd4 c3 Rc1 c2 Kh1 Kh7 Qd2 Qc5 Rxc2 Qxe5 Rc1 f4 Qd3 Qf5 Qxf5 exf5 Rcd1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 d6 Bxf6 exf6 e3 f5 g3 g6 Bg2 Bg7 Ne2 Nd7 c4 Nf6 Nbc3 O-O O-O Re8 Qc2 c6 b4 Be6 Qd3 Ne4 Nf4 d5 cxd5 cxd5 Rfc1 Qd7 Rc2 Bh6 Nce2 Bf8 a3 a5 b5 a4 h4 Rec8 Rac1 Bd6 Nc3 Bxf4 exf4 Rc4 Bf1 Qd6 Qe3 Rcc8 Na2 Rxc2 Rxc2 Rc8 Rxc8 Bxc8 Kg2 Be6 f3 Nf6 Qc3 Bd7 Nb4 Qe7 Qd2 Kg7 Qe2 Qxe2 Bxe2 Kf8 Na2 Ne8 Nc3 Nd6 Kf2 Ke7 Nxd5 Kd8 Nc3 Kc7 Ke3 h6 Kd2 b6 Ke3 Kb7 d5 Kc7 Kd4 f6 Nxa4 Bxb5 Bd1 Bxa4 Bxa4 Kd8 Bc6 Kc7 a4 Kd8 g4 Kc7 g5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O d3 d6 c3 Na5 Bc2 c5 Nbd2 Re8 Nf1 h6 d4 cxd4 cxd4 exd4 Nxd4 Bf8 f3 g6 b3 Bg7 Bb2 Bb7 Qd2 Rc8 Rad1 Qb6 Kh1 Nd7 Ne3 Nc6 Nd5 Qd8 Nxc6 Bxc6 Bxg7 Kxg7 Ne3 Re6 b4 Nb6 Bb3 Re8 Ng4 Qg5 Qd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 e6 e4 d5 exd5 exd5 Nf3 c5 Nc3 c4 Be2 Nf6 O-O Be7 Ne5 O-O Bg5 h6 Bh4 Be6 f4 Bf5 Bf3 Ne4 Bxe7 Qxe7 Nxd5 Qd8 Ne3 Qc8 c3 Nc6 Nxf5 Qxf5 Qc2 Nd6 Qxf5 Nxf5 Bxc6 bxc6 Nxc4 Rab8 Rfe1 Rfe8 Kf2 h5 Rxe8 Rxe8 Re1 Rxe1 Kxe1 f6 Ke2 Kf7 Kd3 Ke6 Ne3 Nd6 c4 g5 fxg5 fxg5 b4 g4 a4 Nf7 b5 cxb5 axb5 Nd6 Nd1 h4 Nc3 Nf5 c5 h3 gxh3 gxh3 Ke4 Ng3 Kf4 Nf1 d5 Kd7 c6 Kc7 Ne4 Nxh2 d6 Kd8 Nc5 Nf1 Nb7 Ke8 d7 Kf7 d8=Q
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 O-O Bd7 c3 Nf6 Re1 g6 d4 exd4 cxd4 Bg7 Nc3 O-O Bg5 Ne7 Bc2 Ne8 e5 dxe5 dxe5 f6 Bb3 Kh8 Bh4 Nf5 e6 Bxe6 Bxe6 Qxd1 Raxd1 Nxh4 Nxh4 Nd6 Nd5 Nb5 a4 c6 Ne7 Na7 Rd7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nbd7 Nge2 c6 Qc2 Re8 O-O Nf8 f3 a5 Kh1 b5 e4 dxe4 fxe4 Ng4 Bd2 Qc7 e5 Nxe5 dxe5 Qxe5 Ng3 Bg5 Rae1 Qc7 Rxe8 Be7 Bxh7 Kh8 Rxf7 Bb7 Rxa8 Bxa8 Nf5 Nxh7 Nxe7 Qd8 Qg6 Qd4 Rf4 Qd8 Qxh7 Kxh7 Rh4
Result: 1-0
Processing game...
FEN: 
Moves: e4 g6 d4 Bg7 Nc3 d6 Be2 Nf6 Nf3 O-O O-O c6 a4 Nbd7 Bf4 Qa5 h3 e5 dxe5 dxe5 Be3 Rd8 Qe1 Qc7 Bc4 h6 Qe2 Nf8 Rfd1 Be6 Rxd8 Rxd8 Rd1 Rxd1 Nxd1 Nxe4 Bxe6 Nxe6 Bxh6 Bxh6 Qxe4 Bg7 Nc3 Qd6 g3 a6 Kg2 b5 axb5 axb5 h4 f5 Qe2 e4 Nd2 Nd4 Qd1 c5 Nb3 Qc6 Nxd4 cxd4 Ne2 d3 cxd3 exd3 f3 dxe2 Qxe2 Kf7 Qd2 Ke6 Qe3 Kd7 Qd3 Kc7 Qb3 Qc4 Qe3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 b5 a4 c6 b3 e5 axb5 cxb5 bxc4 b4 Bb2 exd4 Nf3 Nc6 Nbd2 Nf6 c5 Be7 Bb5 Bd7 Ra6 Rc8 Qa4 Qc7 Nxd4 Nxd4 Bxd4 Bxc5 Bxc5 Qxc5 Bxd7 Nxd7 Rxa7 Qd4 Ke2 Rd8 Qa1 Nb8 Qxd4 Rxd4 Rc1 O-O f3 Rfd8 Nf1 Nd7 Ne3 Ne5 Nd5 b3 Rb7 b2 Rc3 f5 Ke3 Rd1 Rxb2 fxe4 fxe4 Ng4 Kf4 Nxh2 Rc7 Rf8 Kg5 Rd3 Ne7 Kh8 Nf5 Nf1 Rf7 Rdd8 Rxf8 Rxf8 e5 g6 Nd6 Ne3 e6 Nf5 Nf7 Kg8 Rb7 Ng7 e7 Ra8 Nh6 Kh8 Rd7 Ra5 Kf6 Ne8 Kf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O Re1 d6 c3 Bg4 h3 Bh5 g4 Bg6 Be3 Qd7 Nbd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 Bd6 e4 O-O Bd3 Re8 Nge2 a6 a4 Bc7 O-O d6 Bg5 Nbd7 f4 Bb6 Kh1 c4 Bxc4 Qc7 Bd3 Ng4 e5 Nf2 Rxf2 Bxf2 Ne4 Bc5 N2g3 dxe5 f5 f6 d6 Qb6 Bc4 Kh8 Bh4 Qd8 Nh5 b5 Bd5 Ra7 axb5 axb5 Rxa7 Bxa7 Qg4 Rg8 h3 g5 Bxg8 Qxg8 Nhxf6 Qc4 Kh2 Nxf6 Nxf6 Qxg4 hxg4 gxh4 d7 Bxd7 Nxd7 Bd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 c5 Ngf3 Nf6 exd5 exd5 Bb5 Bd7 Bxd7 Nbxd7 O-O Be7 Re1 O-O Nf1 Re8 dxc5 Nxc5 Be3 a6 c3 Rc8 Qc2 Qc7 Rad1 b5 Bd4 Nce4 Ne3 Rcd8 a3 Bf8 Qb3 Qc6 Ne5 Qc7 Nd3 Bd6 g3 Qc8 Bxf6 Nxf6 Nb4 Bxb4 axb4 h6 Rd4 Re5 Red1 Qh3 Qc2 Rde8 Rh4 Qe6 Rhd4 Qh3 Rh4 Qe6 Qd3 Kf8 Rd4 Qh3 Qf1 Qe6 R1d3 Re4 Qd1 Rxd4 Rxd4 Rd8 Kg2 Rd7 Qd3 g6 h3 Kg7 Qd1 h5 Qf3 Ne4 Nxd5 Ng5 Nf4 Qb6 Qd3 Qc6 f3 Rxd4 Qxd4 Kh7 Qd5 Qf6 h4 Ne6 Nxe6 fxe6 Qb7 Kh6 Qxa6 Qe5 Qb6 Qe2 Qf2 Qd1 Qe3 Kg7 Kf2 Qh1 Qxe6 Qh2 Ke3 Qxg3 Qe7 Kg8 Qg5 Qe1 Kd4 Kf7 Kc5 Qf2 Kxb5 Qxf3 c4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O a4 b4 d3 d6 a5 Be6 Nbd2 Rb8 Bc4 Qc8 Nf1 Re8 Ne3 Nd4 Nxd4 exd4 Nd5 Nxd5 exd5 Bd7 Bd2 Bf6 Rxe8 Bxe8 Qe2 Bb5 Re1 Bxc4 dxc4 h6 b3 c5 Bf4 Qd7 h3 Rd8 Qe4 h5 Re2 g6 Qf3 Bg7 Re4 Bf8 Qe2 Qc7 Bg5 Rc8 g4 hxg4 Bf6 gxh3 Qg4 Ra8 Qxh3 Bg7 Bxg7 Kxg7 Rh4 Rg8 Rh7 Kf8 Qg4 Ke8
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Be7 d4 Na5 Nxe5 Nxc4 Nxc4 d5 exd5 Qxd5 Ne3 Qd8 O-O Nf6 c4 O-O Nc3 c6 Rb1 Re8 Bb2 Qc7 Qf3 Bd7 Ne2 Rad8 Ng3 Bc8 d5 cxd5 cxd5 Qa5 Rfd1 Qa6 Rbc1 Ba3 Bxf6 Qxf6 Qxf6 gxf6 Rc4 b5 Rf4 Bb2 Rb4 Be5 Rxb5 Re7 Ngf5 Red7 h3 Kf8 Rb4 Rc7 g4 Rc5 d6 Be6 f4 Bc3 Rb7 Ba5 Rxa7 Bxf5 gxf5 Bb6 Ra6 Rb5 Kf2 Rd7 Kf3 Bxe3 Kxe3 Rxf5 a4 Rh5 Ke4 Rxh3 a5 Rh2 Rc6 Ra2 a6 Ra5 Rc7 Ke8 Rd5 Rxa6 Rc8 Rd8 Rxd8 Kxd8 Kf5 Kd7 Kxf6 Rxd6 Rxd6 Kxd6 Kxf7 h5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 f4 exf4 Nf3 Be7 Bc4 Bh4 Kf1 d5 Bxd5 Nf6 Bb3 Bg4 d3 O-O Qd2 Nh5 Nc3 Nc6 Nxh4 Qxh4 Qf2 Qd8 Bxf4 Be6 Re1 Nxf4 Qxf4 Bxb3 axb3 Nd4 Qf2 f5 e5 Qd7 h4 f4 e6 Nxe6 Re4 Rae8 Rh3 f3 Rxf3 Rxf3 gxf3 Rf8 Ne2 Qc6 Qe3 Nc5 Nd4 Qd7 Re5 Qh3 Ke2 b6 b4 Na4 Re7 Qh2 Kd1 Qxh4 Qe6 Kh8 Re8 Qf4 Qe7 Nxb2 Ke2 Kg8 Qe6 Qf7 Rxf8 Kxf8 Qc8 Qe8 Ne6 Kf7 Qxe8 Kxe8 Nxc7 Kd7 Nd5 Kc6 c4 a5 b5 Kc5 Kd2 Nxc4 dxc4 Kxc4 Nxb6 Kxb5 Nd5 Kc5 Ne3 g5 Kc3 h5 Nf5 h4 f4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qb6 Be3 Qb7 Qg3 b4 Na4 Nbd7 f3 O-O Rad1 Kh8 Rfe1 Rb8 Kh1 Ne5 Ne2 Qc7 c3 Bd7 Nf4 Bxa4 Bxa4 bxc3 bxc3 Qxc3 Bd4 Qa5 Bb3 Qb5 Ne2 Rfc8 Nc3 Qa5 f4 Nh5 Qe3 Nc6 Ne2 Nxd4 Nxd4 Qc3 Rd3 Qc5 Qf3 Nf6 f5 e5 Ne2 Qb4 Nc3 d5 Bxd5 Nxd5 exd5 e4 Qxe4 Rxc3 Rd4 Qd6 f6 Bxf6
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 Nc6 Qd2 e6 O-O-O Bd7 f4 h6 Bh4 g5 fxg5 Ng4 Nf3 hxg5 Bg3 Be7 Be2 Nge5 Kb1 b5 Rdf1 Rc8 Nxe5 Nxe5 Rf2 f6 Rhf1 Bc6 a3 Bb7 h3 Nc4 Bxc4 Rxc4 Qd3 e5 Re2 Qc8 Rf5 Rxc3 bxc3 Qe6 Kb2 Kd7 Rf1 Qc4 Qxc4 bxc4 Ka2 Bc6 Rb1 Bd8 Rb8 Re8 Bf2 Ba5 Rxe8 Kxe8 Kb2 Kf7 Ba7 Ke6 g4 Bd8 Kc1 Be7 Re3 d5 exd5 Kxd5 Kb2 Ke6 Bb6 Bd6 h4 gxh4 Rh3 e4 Rxh4 Bf4 Rh3 Bg5 Bd4 a5 Rh2 a4 Rh1 Bd7 Rh2 Kd5 Rh5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 f4 exf4 Bc4 Qh4 Kf1 b5 Bxb5 Nf6 Nf3 Qh6 Nc3 g5 d4 Bb7 h4 Rg8 Kg1 gxh4 Rxh4 Qg6 Qe2 Nxe4 Rxf4 f5 Nh4 Qg3 Nxe4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O a4 Bb7 d3 Re8 Nbd2 Bf8 c3 h6 Ba2 d6 Nh4 Qd7 Ng6 Ne7 Nxf8 Kxf8 f3 Rad8 b4 Ng6 Nb3 Bc8 Bb1 Nh5 axb5 axb5 Be3 Nh4 Ra2 Re6 d4 Rg6 Kh1 Re8 dxe5 Rxe5 g4 Rf6 Bd4 Ng3 hxg3 Nxf3 Bxe5 Qxg4 Rh2 Nxe1 Qxe1 dxe5 Nd2 Rd6 Bc2 Be6 Kg1 Kg8 Nf1 Qg5 Qe3 Qd8 Rd2 c6 Rxd6 Qxd6 Qc5 Qxc5 bxc5 h5 Nd2 Kf8 Kf2 Ke7 Bb3 Bd7 Nf3 Kf6 c4 bxc4 Bxc4 Be6 Be2 Bg4 Bd1 g6 Ba4 Bd7 Ne1 Ke6 Bb3 Ke7 Nd3 f6 Nb4 f5 Ba4 fxe4 Bxc6 Bxc6 Nxc6 Ke6 Ke3 g5 Kxe4 h4 gxh4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 c4 e6 d4 Nf6 Nc3 dxc4 Qa4 Nbd7 e4 a6 Bxc4 c6 Qd1 Be7 O-O O-O a4 b6 Bd3 Bb7 e5 Nd5 Nxd5 cxd5 Bd2 a5 Rc1 Nb8 Bb5 Na6 Qb3 Rc8 Rc3 Nc7 Bd3 Bb4 Bxh7 Kxh7 Ng5 Kg8 Rh3 Re8 Qf3 Qd7 Qh5 Kf8 Nh7 Ke7 Bg5 f6 Nxf6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 Nf6 e5 Nd5 g3 d6 exd6 e6 Bg2 Bxd6 Nf3 O-O O-O Nc6 d3 b6 Nbd2 Bb7 Qe2 Bc7 Nc4 Re8 a4 h6 Bd2 Qd7 Rad1 Rad8 Bc1 e5 Rfe1 f5 Nh4 Qf7 Bf3 Rf8 Bh5 Qd7 Ng6 Rfe8 Qf1 Nf6 Bf3 Kh7 Nh4 g5 Ng2 f4 Qe2 Kg7 Qc2 Qh3 Nd2 e4 Rxe4 Nxe4 Bxe4 Rxe4 dxe4 Ne5 Ne1 Ba6 c4 Rxd2 Rxd2 Bxc4 Rd3 Bxd3 Nxd3 Nf3 Kh1 Qf1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Nb3 Qa3 Bxf6 gxf6 Be2 Nc6 O-O Bd7 Kh1 h5 Nd1 Rc8 Ne3 Qb4 c3 Qxe4 Bd3 Qa4 Nc4 Rc7 Nb6 Qa3 Rae1 Ne7 Nc4 Rxc4 Bxc4 h4 Bd3 f5 Be2 Bg7 c4 h3 g3 d5 Bf3 dxc4 Re3 c3 Rxc3 Bxc3 Qxc3 O-O Rg1 Rc8 Qf6 Bc6 Bxc6 Rxc6 g4 Ng6 gxf5 exf5 Qxf5 Qxa2 Qxh3 Qc2 f5 Rc3 Qg4 Rxb3 fxg6 Qc6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 c5 dxc5 g5 Bg3 Ne4 e3 Qa5 Be5 O-O Bd3 Nc6 Bxe4 Nxe5 Bxd5 Bg4 Nf3 Bxf3 Bxf3 Nxf3 gxf3 Rac8 O-O
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 f4 Nc5 f5 Be7 Qf3 O-O Be3 e5 Nde2 b5 Bd5 Rb8 b4 Ncd7 O-O Nxd5 Nxd5 Bb7 Nec3 Nf6 Rad1 Nxd5 Nxd5 Bxd5 Rxd5 Rc8 Qg4 f6 Rf3 Rxc2 Rh3 Rf7 Qh5 h6 Qg6 Kf8 Bxh6 gxh6 Rxh6 Qb6 Rc5 Bd8 Rh8 Ke7 Rh7 Rxh7 Qxh7 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O a4 Bb7 d3 d6 Nbd2 Nd7 c3 Nc5 axb5 axb5 Rxa8 Bxa8 Bc2 Bf6 b4 Ne6 Nf1 Bb7 Ne3 g6 Bb3 Bg7 h4 Bc8 h5 Kh8 Nd5 g5 Ne3 Nf4 g3 Nxh5 Nf5 Bxf5 exf5 Qd7 Bxg5 h6 Nh4 Nf6 Bxf6 Bxf6 Qh5 Kh7 Ng2 Ne7 Ne3 Ng8 d4 exd4 cxd4 Bxd4 Ng4 Kg7 Nxh6 Bf6 Bxf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 f4 Nc5 e5 dxe5 fxe5 Nfd7 Bf4 b5 Qg4 h5 Qg3 h4 Qg4 g5 O-O-O Qe7 Nc6 Nxb3 axb3 Qc5 Ne4 Qxc6 Bxg5 Bb7 Rd6 Bxd6 Nxd6 Kf8 Rf1 Nxe5 Qxe6 Qd5 Rxf7 Nxf7 Be7 Kg7 Qf6 Kh7 Nxf7 Qh5 Ng5 Kg8 Qe6 Kg7 Qf6 Kg8 Qe6 Kg7 Bf6 Kh6 Nf7 Kh7 Ng5 Kh6 Bxh8 Qg6 Nf7 Kh7 Qe7 Qxg2 Be5 Qf1 Kd2 Qf2 Kd3 Qf3 Kd2 Qf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 c5 dxc5 g5 Bg3 Ne4 e3 Qa5 Ne2 Bf5 Be5 O-O Nd4 Bg6 Nb3 Nxc3 Bxc3 Bxc2 Nxa5 Bxc3 bxc3 b6 Kd2 bxa5 Kxc2 Rc8 h4 Nd7 hxg5 Nxc5 gxh6 Ne4 c4 Nxf2 Rh4 f5 Rd4 dxc4 Bxc4 Kh7 Rf1 Ng4 Kd2 Rab8 Rxf5 Rb2 Kd3 Rxg2 Be6 Rc7 Rxa5 Nf2 Ke2 Rh2 Kf3 Nh1 Rd7 Rxd7 Bxd7 Kxh6 Rxa7 Kg5 Ra5 Kf6 Bc6 Rc2 Rf5 Ke7 Bd5 Kd6 Rh5 Rd2 Rxh1 Rxd5 a4 Ra5 Ra1 Ke5 e4 Ke6 Ke3 Kd6 Kd4 Kd7 Kc4 Kc6 Kb4 Re5 Rc1 Kb6 Rc4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 f4 Nc5 Qf3 b5 f5 Bd7 fxe6 fxe6 Bg5 Be7 O-O-O O-O e5 Nfe4 Bxe7 Qxe7 Nxe4 Rxf3 exd6 Nxb3 Nxb3 Qf8 gxf3 Qxf3 Nec5 Bc6 Rhe1 e5 d7 Rd8 Rd6 a5 a3 a4 Nd2 Qg2 c3 Bd5 Nd3 Bb3 Nxe5 Qxh2 Nc6 Qxd6 Re8 Kf7 Nxd8 Kg6 Ne6 Qh2 Nf4 Kh6 Nd3 Qg1 Re1 Qg5 Ne5 g6 Rf1 Be6 Nf7 Bxf7 Rxf7 Qd5 Re7 Qd6 Rf7 Qd3 Ne4 Qe3 Nd2 Qd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nxc6 Qf6 Qd2 dxc6 Nc3 Be6 Na4 Rd8 Bd3 Bd4 O-O Ne7 c3 b5 cxd4 Qxd4 Qc2 Qxa4 Qxa4 bxa4 Bc2 Bc4 Re1 Bb5 Be3 Nc8 Bc5 Nb6 Rad1 Rxd1 Rxd1 a6 f4 Nd7 Ba3 h5 Kf2 Rh6 e5 c5 Bf5 Rb6 Rd2 g6 Bc2 Re6 Kg3 Nb6 Bxc5 Nc4 Rd5 Nxb2 f5 Bc6 Rd2 gxf5 Kf4 Nc4 Re2 f6 Bxf5 Rxe5 Bd3 Bd5 Bd4 Rxe2 Bxe2 Ke7 Bxh5 Bxg2 Bd1 a3 h4 Bd5 h5 Ne5 h6 Bxa2 Bc5 Kf7 Bc2 Bc4 h7 Kg7 Bf8 Kh8 Be7 Bd3 Bxf6 Kxh7 Bxe5 Bxc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nc6 f4 Be7 Be3 O-O Qf3 Nxd4 Bxd4 b5 Bxf6 Bxf6 e5 Bh4 g3 Rb8 gxh4 Bb7 Ne4 dxe5 Rg1 g6 Rd1 Bxe4 Qxe4 Qxh4 Ke2 Qxh2 Rg2 Qxf4 Qxf4 exf4 Kf3 Rfd8 Rxd8 Rxd8 Kxf4 Kf8 Ke3 Ke7 c4 h5 a4 bxa4 Bxa4 h4 c5 Rh8 Rc2 h3 Bc6 e5 Kf2 h2 Rc1 a5 Bd5 Rd8 Bg2 Rd2 Kg3 Kd7 Ra1 f5 Kxh2 Rxb2 Rxa5 e4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 Nf3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qe2 Bg6 e4 O-O Bd3 Qa5 e5 Nd5 Bxg6 fxg6 Ne4 c5 Nd6 Qa6 Qxa6 bxa6 Ng5 cxd4 Nxe6 Rfb8 Rd1 Nxe5 Rxd4 Rb6 Rxd5 Rxd6 Rxd6 Bxd6 Bf4 Re8 Nd4 Bc5 Nb3 Bb4 Be3 Nd3 Rb1 Rc8 Bxa7 Rc2 Bd4 Kf7 h3 Be7 Rd1 Nxb2 Na1 Nxd1 Nxc2 Bf6 Bxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nc6 Be3 Be7 f4 O-O O-O Nxd4 Bxd4 b5 e5 dxe5 fxe5 Nd7 Ne4 Bb7 Nd6 Bxd6 exd6 Qg5 Qe2 e5 Bc3 Qg6 Rad1 Kh8 Bd5 Bxd5 Rxd5 Qe6 Rfd1 Rfc8 Ba5 Rc6 b3 Rac8 Bc7 Re8 c4 bxc4 bxc4 f5 h3 h6 Qc2 e4 Qa4 Rc5 Rxc5 Nxc5 Qc6 Nd7 Qd5 Qg6 Qd2 Re5 Qe3 Qe6 Rc1 Rc5 Rc2 Kg8 a4 Kf7 Qf2 e3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nbd7 Nge2 Re8 O-O Nf8 b4 a6 a3 c6 Qc2 g6 f3 Ne6 Bh4 Nh5 Bxe7 Rxe7 Qd2 b6 Rad1 Bb7 Bb1 Nhg7 e4 Rc8 Ba2 Rd7 Nf4 Nxf4 Qxf4 Ne6 Qe5 Re7 Qg3 Qc7 Qh4 Ng7 Rc1 Qd8 Rfd1 Rcc7 Na4 dxe4 fxe4 Qe8 Nc3 Rcd7 Qf2 Ne6 e5 c5 bxc5 bxc5 d5 Nd4 Ne4 Qd8 Nf6 Kg7 Nxd7 Rxd7 Rxc5 Ne6 Rcc1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 Nc6 Nxc6 Qxc6 Re1 Bb7 a3 Rd8 f3 O-O Bh6 Ne8 Kh1 Kh8 Bg5 Bxg5 Qxg5 Nf6 Rad1 Rd7 Rd3 Rfd8 Red1 Qc5 Qe3 Kg8 Kg1 Kf8 Qf2 Ba8 Ne2 g6 Nd4 Qe5 Re1 g5 c3 Kg7 Bc2 Rg8 Nb3 Kf8 Rd4 Ke7 a4 h5 axb5 axb5 Rb4 h4 Nd4 g4 Rxb5 d5 Qxh4 Qh5 Nf5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nxc6 Qf6 Qd2 dxc6 Nc3 Be6 Na4 Rd8 Bd3 Bd4 O-O a6 Nc3 Ne7 Ne2 Bb6 Qf4 Ng6 Qxf6 gxf6 Ng3 h5 Be2 h4 Nf5 Bxf5 exf5 Ne5 Re1 Kf8 Bf4 Rd4 g3 Kg7 Rad1 Re4 Kg2 hxg3 hxg3 Bxf2 Kxf2 Rh2 Kf1 Rexe2 Rxe2 Rh1 Kf2 Rxd1 b3 Rd7 Rd2 Rxd2 Bxd2 c5 Ke3 c6 Ke4 c4 b4 b5 Bf4 Nd7 Kd4 Kf8 Bc7 Ke7 g4 Kf8 Bd6 Kg7 Bc7 Kf8 a3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rad1 Bd7 Nf3 a5 a4 b4 Ne2 Nc6 Nf4 Bf6 Nd3 e5 Be3 Be7 Nd2 Nf6 f3 Rfe8 Kh1 Be6 Rfe1 Rac8 Qf2 d5 Bb6 Qb8 Bc5 Bxc5 Nxc5 Nd4 Nxe6 fxe6 exd5 Nxb3 Nxb3 exd5 Nxa5 Qa8 Nb3 Qxa4 Ra1 Qc6 Re2 d4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 Bxc6 bxc6 d4 exd4 Qxd4 Nf6 O-O Be7 e5 c5 Qd3 dxe5 Qxd8 Bxd8 Nxe5 Be7 Re1 O-O Bg5 Be6 Nd2 Rfe8 h3 h6 Bh4 Rad8 Ndf3 g5 Bg3 Bd5 Rad1 Kg7 c4 Bb7 Rxd8 Rxd8 Nc6 Bxc6 Rxe7 Rd1 Kh2 Ne4 Rxc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rad1 Bd7 a3 Nc6 Nxc6 Bxc6 Bf4 Qb7 Rfe1 a5 e5 dxe5 Bxe5 Bf6 Rd4 Rd8 Rxd8 Bxd8 Ne2 a4 Ba2 b4 axb4 Qxb4 Bc3 Qb7 Nd4 Nf6 Nxc6 Qxc6 Rd1 Be7 h3 Ra8 Rd4 Ne8 Qd3 Bf6 Rc4 Qa6 Bxf6 Nxf6 Qd2 h6 Rd4 Qb6 c3 a3 bxa3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g6 Ba3 Qg5 g3 Nc3 Nxc3 Bxa3 Ne4 Qe7 Nf6 Kf8 Bg2 Bb4 Kf1 Rd8 Qb2 Ba3 Qc3 Bb4 Qb2 Ba3 Qc3 Bb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 cxd4 Nxd4 Bc5 Qd2 O-O O-O-O a6 h4 Nxd4 Bxd4 b5 Rh3 b4 Na4 Bxd4 Qxd4 f6 Qxb4 fxe5 Qd6 Qf6 f5 Qh6 Kb1 Rxf5 Rf3 Rxf3 gxf3 Qf6 Bh3 Kf7 c4 dxc4 Nc3 Qe7 Qc6 Rb8 Ne4 Nb6 Ng5 Kg8 Qe4 g6 Qxe5 Rb7 Rd6 c3 Bxe6 Bxe6 Rxe6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 c3 b6 Bxf6 Bxf6 e4 Bb7 Bd3 c5 d5 e6 Bc4 O-O O-O Na6 Qd3 Nc7 d6 Ne8 Nbd2 Bg7 h4 a6 a4 Qb8 e5 f6 h5 fxe5 hxg6 h6 Rfe1 Qxd6 Qxd6 Nxd6 Nxe5 Bxe5 Rxe5 Rf4 Bd3 Raf8 f3 a5 Kf2 Kg7 Rh5 Ne8 Kg3 Nf6 Re5 Nd5 Be4 R4f6 Nc4 Nf4 Bxb7 Rxg6 Kh2 Rxg2 Kh1 d5 Nxb6 Rb8 Rxe6 Rxb7 Rd6 Rg5 Rd1 d4 Nc4 Kh7 Re1 Rh5 Kg1 Rg7
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 Qf3 g6 Be3 Bg7 h3 e5 fxe5 dxe5 Bh6 Bxh6 Qxf6 O-O Nd5 Qa5 b4 Qd8 Ne7 Qxe7 Qxe7 exd4 Bc4 Nc6 Qc5 Be3 Rf1 Nd8 Rf3 Be6 Rxe3 dxe3 Bxe6 Nxe6 Qxe3 a5 b5 Rac8 O-O-O Rc5 Rd5 b6 Qg3 Rc7 Qd6 Rfc8 Rd2 Rb7 g4 Nc5 Qf6 h6 e5 Re8 h4 Kh7 h5 g5 Rd6 Re6 Qd8 Kg7 a3 a4 Kb2 Rbe7 Rxb6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 Re1 Bg4 c4 Nf6 Nc3 dxc4 Bxc4 O-O d5 Na5 Bd3 c6 h3 Bh5 Re5 Bg6 Bg5 Bd6 Re2 Bb4 Bxf6 gxf6 Rc1 Rc8 Ne4 f5 Ng3 Qxd5 a3 Bd6 Nxf5 Rcd8 Re5 Bxe5 Ne7 Kg7 Nxd5 Bxb2 Nf4 Bxd3 Nxd3 Bxc1 Qxc1 Rxd3 Qg5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 e6 Nf3 a6 d4 cxd4 Nxd4 d6 g4 b5 a3 h6 Bg2 Bb7 O-O Nd7 f4 Rc8 Be3 g5 Qe2 gxf4 Rxf4 e5 Rf5 exd4 Bxd4 Ne5 Nd5 Bg7 Raf1 Rh7 Kh1 Bh8 c3 Ne7 Bxe5 dxe5 Qf3 Nxf5 Qxf5 Rg7 Nf6 Kf8 Nd7 Kg8 Nf6 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 Nc6 g3 g6 Bg2 Bg7 d4 cxd4 Nxd4 O-O Nc2 d6 O-O Be6 b3 Qd7 Bb2 Bh3 Ne3 Bxg2 Kxg2 Ne4 Qc2 Nxc3 Bxc3 Bxc3 Qxc3 Qe6 Nd5 Rac8 Rad1 Rfe8 Rfe1 Qe5 Qd2 Qg7 e4 Ne5 Qe3 Rc5 f4 Ng4 Qf3 h5 h3 Nf6 Nc3 b6 e5 dxe5 fxe5 Nh7 Qe3 Nf8 Nd5 Ne6 a4 Kh7 Re2 Kg8 h4 Rcc8 Rf1 Nc7 Nf4 Red8 e6 f6 Qe4 f5 Qb7 Ne8 Qxa7 Rd6 Qb7 Rcd8 Nd5 Nf6 Qxe7 Nxd5 Qxg7 Kxg7 cxd5 Rxd5 Rc1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 a6 Qd2 Nbd7 Nh3 c5 d5 b5 cxb5 axb5 Bxb5 Ba6 O-O Qa5 a4 Rfb8 Ra3 Nb6 Qe2 Nfd7 Bd2 Bxb5 Nxb5 Qa6 b3 Nc8 Bc1 Na7 Nc3 Qb6 Nb1 Ne5 Nf2 Nc8 f4 Nd7 Nd2 Qa6 Qxa6 Rxa6 Nd3 f5 e5 Ncb6 Nc4 Nxc4 bxc4 Rba8 Bb2 Nb6 Rc1 Nxa4 Rb3 Nxb2 Rxb2 Ra3 Rd2 Rb3 Kf1 Raa3 Ke2 dxe5 Nxe5 Bxe5 fxe5 Re3 Kf2 Rxe5 Re2 Rxe2 Kxe2 Kf7 Kf2 Rd3 Ra1 Rc3 Ra7 Rxc4 d6 Rf4 Kg3 Re4 Rc7 Ke6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 d5 d4 Bb4 Bg5 h6 Bxf6 Qxf6 e3 O-O Rc1 c6 Bd3 Nd7 O-O Bxc3 Rxc3 dxc4 Bxc4 e5 Bb3 exd4 exd4 Qd6 Re1 Nf6 Ne5 Bf5 Rf3 Bh7 Nxf7 Rxf7 Bxf7 Kxf7 Qb3 Kf8 Qxb7 Rd8 Qxa7 Qxd4 Qxd4 Rxd4 Rc3 Be4 f3 Bd5 Kf2 Ne8 b3 Rd2 Re2 Rd1 Rcc2 Nd6 Re1 Rd4 Ke2 Nb5 Ke3 Rb4 Rd2 Ke7 Kf2 Kf7 Re3 Nd6 Ke2 Nf5 Re5 Nd6 Kd3 Nb5 Kc2 Na3 Kc3 Rb7 Kd4 Nb5 Kc5 Kf6 Re3 Nc7 Kd6 Nb5 Kc5 Nc7 a4 Na6 Kd6 Rb8 Rxd5 cxd5 Kxd5 Rb6 Kc4 Rb8 Kd5 Rb6 Kc4 Rb8 Re4 Nb4 a5 Nc6 a6 Rb6 Kc5 Rxa6 Kb5 Ra3 Kxc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 e4 c6 Rb1 a5 h3 a4 Be3 exd4 Nxd4 Re8 Qc2 Nc5 Rfe1 Nfd7 b4 axb3 axb3 Qe7 Rbd1 h5 b4 Ne6 Nxe6 Qxe6 Ne2 Ne5 c5 Ra2 Qc1 dxc5 bxc5 Qb3 Nd4 Qc4 Qb1 Ra3 Re2 Rd8 Nc2 Rxd1 Qxd1 Rd3 Qb1 Bf8 Ne1 Rb3 Qd1 Be6 Kh2 Rb1 Qxb1 Qxe2 Bd4 Nc4 Nf3 Bg7 Qxb7 Bxd4 Nxd4 Qxf2 Nxe6 fxe6 Qxc6 Ne3 e5 Nf1 Kh1 Nxg3 Kh2 Nf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Nbd7 Be2 e5 fxe5 dxe5 Nf5 Qb6 Nd5 Nxd5 Qxd5 Qc5 Qb3 Nf6 Bc4 Qb4 Qxb4 Bxb4 c3 Bf8 Bd3 Be6 Rf1 O-O-O Bc2 Ne8 Bb3 Rd7 Bg5 Bc5 Ke2 Bxb3 axb3 f6 Bc1 Nd6 Nxd6 Rxd6 b4 Ba7 b5 Rhd8 Ra2 Kb8 bxa6 bxa6 b3 Kb7 Rd2 Kc6 Rxd6 Rxd6 g4 Rd7 Bd2 Bb6 Ra1 Kb7 h4 Rd8 h5 Rd6 Be1 Bc7 b4 Bb6 Ra2 Rc6 Ra1 Rd6 Bd2 Rd7 Rf1 Kc6 c4 Bd4 h6 Kb7 c5 a5 Rc1 Rc7 c6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f3 O-O Bg5 a6 Qd2 Nbd7 Nge2 c5 Bh6 Bxh6 Qxh6 b5 h4 b4 Nd5 e5 Nxf6 Qxf6 dxe5 dxe5 Rd1 Rd8 Qg5 Bb7 Nc1 Kg7 Nd3 Rac8 Qxf6 Kxf6 g3 Nf8 Bh3 Rc7 Nf2 Rxd1 Kxd1 h5 Ke2 a5 Nd1 Ne6 Bxe6 Kxe6 Ne3 f5 Nd5 Rf7 b3 fxe4 fxe4 Kd6 Rd1 Kc6 Ne3 Kc7 Nd5 Kc6 Ne3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7 Qg4 Kf8 h4 Qc7 Qd1 cxd4 cxd4 Qc3 Bd2 Qxd4 Nf3 Qe4 Be2 b6 O-O Ba6 c4 Nbc6 Ng5 Qxe5 Re1 Qf6 Bh5 g6 cxd5 exd5 Bg4 h6 Ne6 fxe6 Rxe6 Qf7 Qa4 Bc4 Rxc6 Nxc6 Qxc6 Re8 Bd7 Kg7 Bc3 Kh7 h5 Rhf8 hxg6 Qxg6 Bxe8 Qxe8 Qd6 Qf7 Bd4 Re8 Rc1 Re4 Be5 Rg4 Rc3 Rg6 Qd8 Rg8 Qh4 Rg5 f4 Rh5 Qd8
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 Nc6 d4 cxd4 Nxd4 e6 a3 Be7 e4 O-O Be2 d5 exd5 exd5 O-O Nxd4 Qxd4 Be6 cxd5 Nxd5 Nxd5 Bxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 Bb7 Nc2 Nb8 a4 bxa4 Rxa4 Nd7 Rb4 Nc5 Rxb7 Nxb7 b4 Bg5 Na3 O-O Nc4 a5 Bd3 axb4 cxb4 Qb8 h4 Bh6 Ncb6 Ra2 O-O Rd2 Qf3 Qa7 Nd7 Nd8 Nxf8 Kxf8 b5 Qa3 Qf5 Ke8 Bc4 Rc2 Qxh7 Rxc4 Qg8 Kd7 Nb6 Ke7 Nxc4 Qc5 Ra1 Qd4 Ra3 Bc1 Ne3
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nxc6 Qf6 Qd2 dxc6 Nc3 Be6 Na4 Rd8 Bd3 Bd4 O-O a6 Qa5 b5 Nc3 Bb6 Qxa6 Nh6 Be3 Bc8 Qa3 Bxe3 fxe3 Qe5 Be2 Rd6 Rad1 O-O Rxd6 cxd6 Rd1 c5 Nxb5 Bg4 Qd3 Qxb2 a4 Qb4 Bf3 Qxa4 Nxd6 Qb4 h3 Be6 e5 Qh4 Qe4 Qe7 Rb1 f5 Qc6 Kh8 Qxc5 Rc8 Qxc8 Bxc8 Rb8 Ng8 Rxc8 Qa7 Kh2 Qe7 Re8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 e6 Nd2 h6 Bh4 c5 e3 cxd4 exd4 Be7 Bd3 b6 Ngf3 Bb7 O-O O-O Bxf6 Bxf6 Be4 Qc7 Re1 Be7 Bxb7 Qxb7 Nc4 Nc6 d5 Nb4 Ne3 Bf6 c3 Nxd5 Nxd5 exd5 Re2 b5 Rd2 b4 cxb4 Qxb4 Rb1 Rac8 Rxd5 Qe4 Rd2 Rfe8 h3 d5 b3 Rc5 Qf1 Rec8 Re1 Qf5 Red1 a5 Qd3 Qe4 Re2 Qxd3 Rxd3 a4 Red2 axb3 axb3 Rb5 Rd1 Rc2 Rb1 Kf8 Ne1 Ra2 b4 h5 Kf1 g5 Rdb3 Bd4 Rf3 Kg7 g4 hxg4 hxg4 Rb8 Nd3 Rh8 Kg2 Rh4 Rf5 f6 Kg3 Ra3 Rd1 Be5 Kf3 Rh3 Ke2 Bc3 Nc1 d4 Ra5 Bxb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 c4 Nf6 Nc3 Nc6 d4 cxd4 Nxd4 g6 f3 Bg7 Be3 O-O O-O Rac8 b3 Rfd8 Rc1 e6 Kh1 d5 exd5 exd5 c5 Ne8 Ncb5 a6 Nc3 Nc7 Na4 Qe7 Re1 Ne6 Nb6 Rb8 Nxc6 bxc6 b4 Qh4 Bd2 Re8 Qe2 Nc7 Qd3 Qf2 a4 Bh6 Red1 Re2 Qxe2 Qxe2 Bxh6 Re8 Bf4 Ne6 Re1 Nxf4 Rxe2 Rxe2 b5 Nd3 Rb1 axb5 axb5 cxb5 c6 Rc2 Nxd5 Kf8 c7 Ke8 h3 Kd7 Rxb5 Nc5 Rb8 Ne6 c8=Q
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 c5 exd5 exd5 Ngf3 Nc6 Bb5 Bd6 dxc5 Bxc5 O-O Ne7 Nb3 Bd6 Re1 O-O Bg5 Bg4 c3 Ne5 Be2 N5g6 Nfd4 Bxe2 Qxe2 h6 Bxe7 Nxe7 Rad1 a6 Qf3 Qd7 Nc2 Qa4 Ra1 Rfd8 Ne3 Qf4 Qxf4 Bxf4 Rad1 Rac8 g3 Bxe3 Rxe3 Rd7 Red3 g6 Nd4 Rcd8 Nc2 f5 Ne3 Kf7 h4 g5 hxg5 hxg5 Kg2 f4 Ng4 Ng6 Rh1 Ke6 Kf3 Kf5 Rhd1 Ke6 R3d2 b5 Re2 Kf5 Rde1 Rd6 Rh1 d4 Rh7 Rf8 Rc7 Nh8 cxd4 Rxd4 Re5 Kg6 Rc6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 cxd5 cxd5 Bf4 Nc6 e3 a6 Bd3 e6 Nf3 Be7 h3 b5 O-O O-O a4 b4 Ne2 Bb7 Rc1 Rc8 Ng3 Na5 Qe2 Qb6 Ne5 Rxc1 Rxc1 Nb3 Rc2 Na1 Rc1 Nb3 Rd1 Rc8 Nh5 g6 Nxf6 Bxf6 Nd7 Qd8 Nxf6 Qxf6 Bxa6 Qe7 Qb5 Bxa6 Qxa6 Qe8 Qb7 Ra8 Qxb4 Qxa4 Qe7 Nc5 Rc1 Ne4 Rc7 Qe8 Qxe8 Rxe8 b4 Rb8 f3 Nf6 Rxf7 Kxf7 Bxb8 Ke7 b5 Kd8 Kf2 Kc8 Bd6 Ne8 Be5 Kb7 e4 Kb6 exd5 exd5 Kg3 Kxb5 Kf4 Kc6 Kg5 Kd7 Kh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c6 Nf3 dxc4 a4 Bb4 e3 b5 Bd2 a5 axb5 Bxc3 Bxc3 cxb5 b3 Bb7 bxc4 b4 Bb2 Nf6 Bd3 Nbd7 O-O Qc7 Nd2 O-O f4 a4 e4 a3 Bc1 Rfd8 e5 Ne8 Nb3 g6 Qd2 Bc6 Qxb4 Rdb8 Qc3 Bxg2 Kxg2 Qb7 Rf3 Qxb3 Qxb3 Rxb3 Be4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 d6 d4 cxd4 Nxd4 Nf6 Bc4 Qb6 Nde2 e6 O-O Be7 a3 a6 Ba2 O-O Kh1 Rd8 Ng3 d5 exd5 exd5 Nh5 Be6 Nf4 Rd7 f3 d4 Na4 Qb5 Nxe6 Qxa4 Nf4 Rad8 Re1 Qb5 Qd3 Qxd3 Nxd3 Nd5 Bd2 g6 g3 Kg7 f4 b6 Bb3 Bf6 Bxd5 Rxd5 a4 Rc8 Re2 Rd7 Kg2 Rdc7 g4 Nd8 a5 bxa5 Rxa5 Rxc2 Kf3 Ne6 b4 R2c6 f5 gxf5 gxf5 Nc7 Rg2 Kh8 Rc5 Rxc5 bxc5 Nd5 Bh6 Ne7 Ke4 Re8 Bc1 Ng6 Kd5 Ne7 Kc4 Nc6 Bf4 a5 Kb5 Nb4 Nxb4 axb4 Kxb4 d3 c6 Re2 c7 Rxg2 c8=Q Kg7 Bd6 Rb2 Ka3
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f3 Nc6 Nge2 a6 Bg5 Rb8 Rc1 O-O Qd2 e5 dxe5 Nxe5 Ng3 Be6 b3 Qd7 Be2 Ne8 O-O Nc6 Be3 Ne7 Rfd1 f5 exf5 Nxf5 Nxf5 Bxf5 Bd4 Nf6 Nd5 Rbe8 Bf1 c6 Nxf6 Bxf6 Bc3 Bxc3 Rxc3 Qg7 Re3 Rxe3 Qxe3 Qe5 Qxe5 dxe5 c5 Kf7 b4 Be6 a3 Ke7 Kf2 e4 Re1 exf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 f3 g6 c4 Bg7 Nc3 O-O Bg5 c6 Qd2 Qc7 Bd3 a5 Nge2 e5 d5 Nbd7 Ng3 Nc5 Be2 Bd7 Rc1 Rae8 O-O Bc8 Kh1 Kh8 Bd1 Ng8 a3 Na6 Na4 Bd7 b4 axb4 axb4 f5 exf5 Bxf5 dxc6 bxc6 b5 Nb8 b6 Qd7 Be2 c5 Nc3 Nc6 Bd1 Rb8 Nd5 Nd4 Ra1 Ra8 Rxa8 Rxa8 Ne4 Rf8 Nec3 Be6 Ba4 Qb7 Rb1 Bf5 Ne4 Bxe4 fxe4 Qa6 Qd1 Qxc4 b7 Rb8 Bd7 Qa6 Bc8 c4 Rc1 Nb3 Rb1 Nc5 Qg4 Nxb7 h3 Qc6 Bd7 Qc5 Be3 Qa3 Bc6 c3 Rxb7 Rxb7 Bxb7 Qb3 Qc8 h5 Nxc3 Kh7 Bd5
Result: 1-0
Processing game...
FEN: 
Moves: c4 e6 g3 Nf6 Bg2 d5 d4 Bb4 Nd2 O-O Nf3 c6 O-O Nbd7 Qc2 a6 b3 b6 Bb2 Bb7 e4 Bxd2 Nxd2 dxe4 Nxe4 Nxe4 Qxe4 Re8 Rad1 Rb8 Qf4 b5 d5 cxd5 cxd5 e5 Qd2 Qb6 Rc1 Qd6 Bc3 e4 Rfd1 Nf6 Bxf6 gxf6 Qf4 Qxf4 gxf4 Rbc8 Rxc8 Bxc8 d6 Bd7 b4 Kg7 Rc1 Re6 Rd1 Re8 Rc1 Re6 Rd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 cxd5 exd5 Qc2 Be7 Bg5 Nbd7 e3 O-O Bd3 Re8 h3 Nf8 Bf4 Ne6 Bh2 Bd6 Bxd6 Qxd6 O-O g6 Rab1 a5 a3 Ng7 b4 axb4 axb4 Bf5 Rfc1 Bxd3 Qxd3 Ne4 b5 Ra3 Nd2 f5 bxc6 bxc6 Ncxe4 fxe4 Qc2 Rc8 Rb7 Ne6 Qb2 Ra6 Nb3 Raa8 Nc5 Rab8 Nxe6 Rxb7 Qxb7 Qxe6 Rc2 Qe8 Rc3 Qe6 Rc1 Qe8 Ra1 Rb8 Qc7 Ra8 Rxa8 Qxa8 Qd7 Qa4 Kh2 Qb5 Kg3 Qc4 Qe8 Kg7 Qe7 Kg8 Qf6 Qb5 h4 Qb8 Kg4 Qc8 Kg5 Qd7 Kh6 Qc7 f4 Qf7 Qe5 Qf8 Kg5 Qd8 Qf6 Qd7 g4 Qf7 Qxf7 Kxf7 f5 c5 dxc5 d4 c6 dxe3 fxg6 hxg6 c7 e2 c8=Q e1=Q Qd7 Kf8 Kxg6
Result: 1-0
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 d5 cxd5 Nxd5 g3 g6 Bg2 Nxc3 bxc3 Bg7 h4 h6 Rb1 Nd7 Qc2 Nb6 d3 O-O c4 Bd7 Be3 Rb8 Nh3 Qe8 Qc1 Bc6 O-O Bxg2 Kxg2 Qc6 f3 Kh7 Rb5 e5 h5 gxh5 Rc5 Qg6 Rxc7 Rbc8 Rxb7 Nd5 Nf2 h4 g4 Kg8 Rh1 e4 fxe4 Qa6 exd5 Qxb7 Bxh6 Qe7 Qg5 Qxg5 Bxg5 f6 Bf4 Rfe8 Kf3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 a6 c5 g6 Bf4 Bg7 e3 O-O h3 Nbd7 Bd3 Re8 Bh2 Nh5 g4 Nhf6 Rc1 b6 cxb6 Qxb6 Na4 Qa5 Kf1 Bb7 a3 Ne4 b4 Qd8 Kg2 a5 b5 Rc8 bxc6 Rxc6 Qb3 Qa8 Rxc6 Bxc6 Rc1 Rc8 Nb6 Nxb6 Qxb6 Bb7 Rxc8 Bxc8 Ne5 Bb7 Nd7 Qc8 Qc7 e6 f3 Nc3 Nc5 Qxc7 Bxc7 Bc6 Bxa5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 e6 c3 Nge7 d4 cxd4 cxd4 d5 Nc3 dxe4 Nxe4 Nd5 O-O Bd7 Bg5 f6 Bh4 Qb6 Bc4 Be7 Nc3 Nxc3 bxc3 Na5 Bd3 Qc7 Rc1 Ba3 Rc2 Ba4 Qb1 Bxc2 Qxc2 O-O-O Re1 Rhe8 Bxh7 g5 Bg3 Bd6 Rxe6 Bxg3 Bf5 Bxh2 Nxh2 Rxe6 Bxe6 Kb8 Nf3 Qf4 Nd2 Nc6 g3 Qc7 Qf5 Rf8 Ne4 Nxd4 cxd4 Qc1 Kg2 Rh8 Kf3 Qd1 Ke3 Rh1 Nxg5 Re1 Kf4 fxg5 Kxg5 Qxd4 Qf6 Qc5 Kg6 a6 g4 Qc2 Kg7 Qe4 Bf5 Qf4 Be6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 f5 g3 Nf6 Bg2 g6 d4 Bg7 Nc3 d6 Nh3 Nc6 d5 Ne5 Nf4 O-O b3 Ne4 Nxe4 fxe4 Bxe4 Nf3 exf3 Bxa1 Qd3 Be5 h4 Bxf4 Bxf4 Bf5 h5 Bxe4 fxe4 Qe8 Bh6 Rf6 Bg5 Rf8 f4 e5 dxe6 Qxe6 hxg6 Qxg6 Rh6 Qg7 Qd5 Kh8 Qe6 Rae8 Bf6 Rxf6 Qxf6 Rxe4 Kf2 Qxf6 Rxf6 Kg7 Rf5 b6 a4 a5 Rd5 Kf6 Rd1 Kf5 Kf3 Re7 g4 Kf6 Rh1 Kg6 f5 Kf6 Kf4 Kg7 g5 Re2 Rh3 Re1 f6 Kg8 Re3 Rf1 Kg4 Rg1 Kf5 Rf1 Ke6 Rg1 Ke7
Result: 1-0
Processing game...
FEN: 
Moves: c4 e5 g3 Nc6 Bg2 d6 Nc3 g6 Rb1 a5 a3 Bg7 b4 axb4 axb4 Nf6 e3 O-O Nge2 Ne7 O-O d5 cxd5 Nfxd5 Nxd5 Nxd5 Qc2 Be6 b5 Qd6 Nc3 Nxc3 dxc3 Ra2 Rb2 Rxb2 Qxb2 Bc4 Re1 Qb6 Bf1 Bxf1 Rxf1 c6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 Ne4 Bf4 g5 Bc1 h6 f3 Nf6 e4 d6 Nc3 e5 Nge2 Bg7 dxe5 dxe5 Qxd8 Kxd8 Ng3 Nc6 Be3 Ke8 Nb5 Kd8 O-O-O Bd7 Nf5 Bf8 Bc4 a6 Nc3 Ke8 g4 Na5 Be2 Bb4 h4 Bxc3 hxg5 Bxb2 Kxb2 Bxf5 exf5 Nd7 gxh6 f6 g5 fxg5 Bxg5 Rh7 f4 b6 Bh5 Kf8 Bg6 Nc4 Kc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6 Nf3 Bxa6 Nc3 d6 Nd2 Bg7 e4 Bxf1 Nxf1 Qa5 Bd2 O-O Ne3 Nbd7 O-O Qa6 Qc2 Ne5 b3 Nfd7 f4 Nd3 a3 Bd4 Kh1 Rfc8 Rab1 c4 b4 Nf6 Ned1 Rcb8 Ne3 Rc8 b5 Qb6 Nxc4 Nf2 Rxf2 Rxc4 Rf3 Rxa3 Qb2 Nxe4 Qxa3 Nxd2 Qa2 Be3 Rc1 Qd4 b6 Rxc3
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 g3 e6 Nf3 Nc6 Bg2 d5 cxd5 exd5 d4 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8 Rc1 Bf8 Nxc6 bxc6 Na4 Bd7 Bc5 Bxc5 Nxc5 Bg4 Re1 Qa5 h3 Bf5 Qd4 Rab8 a3 Qb5 b3 Ne4 b4 a5 Nxe4 Bxe4 Rc5 Qb6 bxa5 Qb2 Qxb2 Rxb2 Rxc6 Ra2 Rec1 Rxa3 Rc8 Rxc8 Rxc8 Kh7 f3 Bf5 Rc5 Be6 Kf2 g5 f4 gxf4 gxf4 d4 Be4 Kg7 f5 Ba2 Bd3 Kf6 a6 Ke7 Rc6 f6 Rc7 Kd6 Rh7 Ra4 Rxh6
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f4 O-O Nf3 c5 dxc5 Qa5 Bd3 Qxc5 Qe2 Bg4 Be3 Qa5 O-O Nc6 Rac1 Nd7 Qf2 Bxf3 gxf3 Nc5 Bb1 Na4 Nxa4 Qxa4 b3 Qa3 c5 dxc5 Bxc5 Qxc5 Rxc5 Bd4 Rd1 Bxf2 Kxf2 Rfd8 Rcd5 e6 Rxd8 Rxd8 Rxd8 Nxd8 Ke3 Kf8 f5 e5 f4 f6 fxe5 fxe5 fxg6 hxg6 h4 Kg7 Kd3 Kf6 Kc4 Ke6 Kc5 b6 Kb5 Kd6 b4 Kc7 a4 Ne6 Ba2 Nd4 Ka6 Kb8 Bf7 Nf3 Bxg6 Nxh4 Bf7 Nf3 a5 bxa5 bxa5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 c6 Nc3 d5 Nf3 e6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Bb7 g3 c5 d5 Nxf6 Bg2 Bh6 Bxf6 Qxf6 O-O O-O-O Nxb5 exd5 Nxa7 Kb8 Nb5 Bg7 a4 Qh6 h4 Bf6 Qe1 Bxh4 Qa5 Be7 Qc7 Ka8 Qa5 Kb8 Qc7 Ka8 Rfe1 Bd6 Qb6 Bb8 a5 Rd7 Re8 Qh2 Kf1 Qxg2 Kxg2 d4 Qxb7 Rxb7 Rxh8 Rxb5 a6 Ka7 Rf8 Rxb2 Rxf7 Ka8 a7 c3 Rf8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Be2 Be7 O-O Qc7 Qe1 Nbd7 a4 b6 Bf3 Bb7 Kh1 Rd8 Be3 O-O Qg3 Nc5 f5 e5 Bh6 Ne8 Nb3 Nd7 Rad1 Kh8 Be3 Nef6 Qf2 Rfe8 Rfe1 Bf8 Bg5 h6 Bh4 Rc8 Qf1 Be7 Nd2 Qc5 Nb3 Qb4 Be2 Bxe4 Nxe4 Nxe4 Bxe7 Rxe7 Bf3 Nef6 Qxa6 Ree8 Qe2 Kg8 Bb7 Rc4 Qd2 Qxa4 Qxd6 Rxc2 Nd2 Nf8 Ne4 N8d7 Nxf6 Nxf6 Qxb6 Ng4 Rf1 e4 Bd5 e3 Bb3 Qe4 Bxc2 Qxc2 Rd8 Rxd8 Qxd8 Kh7 Qe7 Qc4
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 Nf6 e3 Bf5 Nge2 O-O Ng3 Be6 Bd3 c5 dxc5 Bxc5 O-O Nc6 Rc1 Bd6 Nge2 Rc8 Bb1 Bxf4 Nxf4 Bg4 f3 Be6 Qd2 Qe7 Nb5 a6 Nd4 Ne5 b3 Bd7 g4 Ng6 Ng2 Ne8 Rxc8 Bxc8 Rc1 Nd6 Bxg6 hxg6 Nf4 Re8 Nxd5 Qg5 Nf4 Qe5 Ng2 g5 Qb4 Nb5 Nxb5 axb5 Qd4 Qe7 Qc5 Qf6 Qxb5 Rd8 Qe2 Be6 Qf2 Rd3 h4 gxh4 Nxh4 g5 Ng2 Qe5 Qc2 Qd6 Ne1 Rxe3 Qf2 Qe5 Ng2 Re2 Re1 Rxf2 Rxe5 Rxf3 Ne3 Bd7 Rxg5 Kf8 Nf5 Bxf5 gxf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Bc4 Ngf6 Ng5 e6 Qe2 Nb6 Bb3 h6 N5f3 a5 c3 c5 a3 Qc7 Ne5 cxd4 cxd4 a4 Bc2 Bd7 Nxd7 Nbxd7 Qd1 Bd6 Ne2 Nd5 Bd2 b5 Nc3 Nxc3 Bxc3 Nf6 Qd3 Nd5 Bd2 Ke7 Rc1 Qc4 Ke2 Rhb8 g3 Qxd3 Bxd3 b4 Ra1 bxa3 bxa3 Rb3 Bc2 Rxa3 Rxa3 Bxa3 Ra1 Bb2 Rxa4 Rxa4 Bxa4 Bxd4 f4 Kd6 Kf3 f5 h4 Bb2 g4 fxg4 Kxg4 Nf6 Kf3 Nd5 Bc2 Bf6 h5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 O-O Nf6 Re1 a6 Bf1 b6 d4 cxd4 Nxd4 Bb7 f3 e6 a4 Be7 a5 b5 c4 bxc4 Bxc4 O-O Nc3 Ne5 Bf1 Qc7 Be3 d5 exd5 Nxd5 Nxd5 Bxd5 Nf5 Bb4 Bb6 Qb8 Qd4 Bxe1 Rxe1 exf5 Qxd5 Re8 Rc1 Re6 b4 h5 Bc7 Qa7 Bb6 Qb8 Bc7 Qa7 Bb6 Qb8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 Qe2 h6 Ne4 Nxe4 Qxe4 Qc7 Qg4 g5 Qh3 Rg8 Nd2 Bf8 Ne4 Bg7 O-O Bxd4 Be3 Be5 Rad1 Nf6 Nxf6 Bxf6 Bh7 Rh8 Qxh6 Be7 Qg7 Rf8 Rd3 Bd7 Bxg5 Bxg5 Qxg5 Qd8 Qe5 Qe7 Qc7 Bc8 Qa5 b6 Qe5 Ba6 Be4 Rc8 c4 Rg8 b3 Bb7 Rfd1 Ba8 c5 bxc5 Rd6 c4 bxc4 c5 Bxa8 Rxa8 Qxc5 Rb8 g3 Qb7 Qd4 Kf8 Qf6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 Nbd7 Bg5 h6 Bh4 g5 Bg3 Nh5 h4 g4 Nh2 Nxg3 fxg3 h5 O-O f5 exf5 Nc5 b4 e4 Rc1 Nd3 Bxd3 exd3 f6 Rxf6 Qxd3 Qf8 Nb5 Bf5 Rxf5 Rxf5 Nxc7 Rc8 Ne6 Qf6 Nf1 Re5 Rd1 Qf5 Qxf5 Rxf5 c5 Bf8 Ne3 Rf6 Nc4 dxc5 b5 Bh6 Re1 Re8 Re5 Re7 Rxh5 Ref7 Kh2 Bc1 Re5 Rf1 Re4 Rd1 Rxg4 Kh7 Ne5 Re7 Nf8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 d6 Nge2 Nc6 d4 cxd4 Nxd4 Nf6 Bc4 Qb6 Nb3 e6 Bf4 Ne5 Be2 Be7 Be3 Qc7 f4 Nc6 Bf3 a6 O-O O-O a4 b6 g4 Rb8 g5 Nd7 Bg2 Re8 Rf3 Nc5 Rh3 g6 Qg4 Nb4 Qh4 h5 Bf3 Bf8 Bxh5 gxh5 Qxh5 Bg7 Bd4 e5 f5 Nxe4 Qh7 Kf8 Nxe4 Bxf5 Qxf5 exd4 Nf6 Qxc2 Nxd4 Qxb2 Rd1 Re5 Nd7 Ke7 Nxe5 Bxe5 Qe4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c5 Nf3 Bg7 Bg2 O-O O-O d6 Nc3 Nc6 d5 Na5 Nd2 e5 b3 Bd7 Bb2 Ng4 h3 Nh6 e3 f5 f4 a6 Qc2 b5 Nd1 Rb8 Bc3 Re8 Rb1 Nf7 Nf2 exf4 exf4 Re3 Bxg7 Kxg7 Rfe1 Rxe1 Rxe1 Qf6 cxb5 axb5 b4 cxb4 Qc7 Qd8 Qxd8 Rxd8 Nd3 Rc8 Nxb4 Kf6 Kf2 Nd8 Bf1 Ndb7 Re3 h6 Re1 g5 Re3 Rc7 Re1 Rc8 Re3 Rc7 h4 g4 Re1 Rc3 Re3 Rc1 Re1 Rc3 Re3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 Bb3 h6 h3 a6 Nbd2 Be6 Bc2 Ba7 Qe2 Qe7 b4 d5 a4 b5 O-O O-O axb5 axb5 d4 exd4 e5 dxc3 exf6 Qxf6 Nb3 Nxb4 Bb1 d4 Rxa7 c2 Rxa8 cxb1=Q Rxf8 Kxf8 Qxb5 Qxb3 Qb8 Ke7 Qxc7 Ke8 Bd2 Qd8 Qe5 Kf8 Nxd4 Nd3 Qe3 Qc4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 b6 a3 Bb7 Nc3 d5 Bg5 Be7 e3 O-O Bd3 Nbd7 cxd5 exd5 O-O c5 Rc1 Ne4 Bf4 a6 Qc2 Ndf6 dxc5 Bxc5 Rfd1 Qe8 b4 Be7 Be2 Rc8 Qb2 b5 Nd4 Nd6 Bd3 Nc4 Qb3 Nh5 Bf5 Ra8 Nde2 Nf6 Bg5 Rd8 Nf4 d4 exd4 h6 Bxf6 Bxf6 Nce2 Be4 Bxe4 Qxe4 Qg3 Rfe8 Qc3 Rd6 Re1 Red8 Rcd1 Bxd4 Nxd4 Qxf4 Ne2 Qe5 Rxd6 Rxd6 a4 Re6 Qc1 Qd6 axb5 axb5 Ng3 Qxb4 Rxe6 fxe6 h3 Qc5 Nf1 Qd5 Qa1 Qe5 Qa7 Kh7 Qd7 Qd5 Qe7 Qd6 Qb7 Qd5 Qe7 Qe5 Qd7 Nd6 Ne3 Nf5 Qd3 Kg8 Qd8 Kf7 Qd7 Kg6 Qd3 Qd4 Qb1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c6 d4 d5 Nf3 Nf6 Qc2 dxc4 Qxc4 Bf5 Nc3 Nbd7 g3 e6 Bg2 Be7 O-O O-O e3 Ne4 Qe2 Qb6 Rd1 Rad8 Ne1 Ndf6 Nxe4 Nxe4 f3 Nd6 a4 Qb3 e4 Bg6 Rd3 Qb4 b3 Nc8 Nc2 Qb6 Bf4 c5 Be3 cxd4 Nxd4 Bc5 Rad1 e5 Nc2 Rxd3 Qxd3 Ne7 b4 Bxe3 Qxe3 Rd8 Rxd8 Qxd8 Bf1 b6 Qc3 f6 Bc4 Bf7 Ne3 Qd4 Bxf7 Kxf7 Qb3 Kf8 Kg2 Qd2 Kh3 Qe2 Ng2 h5 Qe3 Qc4 Qd2 Qe6 g4 hxg4 fxg4 Qc4 Qe1 Qb3 Ne3 Qd3 Kg3 Qxe4 Qd2 Qf4 Kg2 Qd4 Qxd4 exd4 Nc4 Nc6 b5 Ne5 Nd6 d3 Kf2 Nxg4 Ke1 Nxh2 Kd2 Nf3 Kxd3 Ke7 Nf5 Kf7 Ke4 Nd2 Kd5 g5 Nd6 Kg6 Kd4 Nb3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 c5 Bg2 cxd4 Nxd4 O-O O-O Nc6 Nc3 Nxd4 Qxd4 d6 Qd3 a6 Bd2 Rb8 Rac1 b5 b3 Bf5 e4 Bd7 h3 bxc4 Qxc4 Qa5 Qd3 Qa3 Rc2 Bb5 Nxb5 axb5 Rc7 e6 Be3 Rbc8 Ra7 Qb4 Rd1 Ne8 a3 Qc3 Qxb5 Qb2 Qd3 Rc3 Qb1 Qxb3 Qxb3 Rxb3 a4 h5 a5 Ra3 a6 Be5 Bh6 Bg7 Bg5 Bf6 Bxf6 Nxf6 Rxd6 Ra1 Kh2 Ra2 e5 Nh7 Rad7 Rxf2 Rd2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 f5 exf5 Bxf5 d4 e4 d5 exf3 Qxf3 Qe7 Kd1 Be4 Qh3 Qf7 dxc6 Bxc6 Re1 Be7 Bxc6 bxc6 Nd2 Nf6 Nf3 O-O Ng5 Qd5 Bd2 h6 Qe6 Kh8 Nh3 Qxe6 Rxe6 Rfe8 Kc2 Kg8 Rae1 Kf7 f3 Nd7 R6e2 Ne5 Re4 Nd7 Nf4 d5 R4e2 Bd6 Ne6 Bxh2 Nxg7 Rxe2 Rxe2 Kxg7 Re7 Kg6 Rxd7 Bd6 Kd3 h5 c4 Rg8 c5 Be5 b3 Kf5 Rh7 Rh8 Rxh5 Rxh5 g4 Kg6 gxh5 Kxh5 f4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Qb6 a3 Nc6 Nxc6 bxc6 b3 Bb7 Bb2 d5 O-O-O Qa5 e5 Nd7 Na4 Qc7 Re1 g6 g4 c5 Bg2 Rb8 Qd1 c4 Kb1 Qa5 Bc3 Qb5 Ka2 Bc6 f5 Qb7 fxe6 fxe6 Nb2 cxb3 cxb3 Nc5 Bb4 Nxb3 Qxb3 a5 Rhf1 axb4 a4 Be7 Rc1 Rf8 g5 Bxg5 Qh3 Rf5 Rxf5 exf5 Rd1 b3 Kb1 Rc8 Rd3 Bxa4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 g4 h6 f4 Nc6 Be2 e5 Nf5 g6 Ng3 exf4 Bxf4 Be6 Rf1 Rc8 h3 Qb6 Qd2 Bg7 Bxd6 Nxg4 Bxg4 Qxb2 e5 Nxe5 Rb1 Qxc3 Qxc3 Rxc3 Bxe6 fxe6 Rxb7 Nc4 Bb4 Re3 Ne2 Be5 Rff7 Rxh3 Nd4 Re3 Kf1 Re4 Rfe7 Kd8 Nc6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 f4 Be7 O-O Qc7 Qe1 Nbd7 Bf3 O-O Kh1 Kh8 a4 Rb8 g4 b6 g5 Ne8 Bg2 Bb7 b3 Qd8 h4 g6 Bb2 Ng7 Rd1 Rc8 f5 e5 f6 exd4 fxe7 Qxe7 Rxd4 f6 Qd2 fxg5 Rxf8 Rxf8 Rxd6 Ne5 Rxb6 gxh4 Nd5 Bxd5 Qxd5 Re8 Bh3 Qc7 Re6 Rxe6 Bxe6 Nc6 Qg5 Qd6 Bd5 Nb4 Qf6 Qxf6 Bxf6 Nxc2 Bc3 h6 b4 Kh7 b5 axb5 axb5 Nh5 b6 Ng3 Kh2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d6 Nf3 g6 c4 Bg7 Nc3 e5 dxe5 dxe5 Qxd8 Kxd8 Bd2 f6 O-O-O Ke8 h3 Be6 b3 Na6 g4 Rd8 Bg2 Ne7 Be3 Nc8 Nb5 Rxd1 Rxd1 Ke7 Nxe5 c6 Nxc6 bxc6 Bxc6 Rd8 Bb7 Rxd1 Kxd1 Nb8 Bf4 a6 Nd4 Nd6 Bf3 Bc8 c5 Nb7 b4 f5 e3 fxg4 hxg4 a5 Bxb8 axb4 Nc6 Kd7 Be5 Bf8 Nxb4 Nxc5 Bc6 Kd8 g5 Be6 Bd5 Kd7 Bc6 Kc8 Bd5 Kd7 f4 Bxd5 Nxd5 Ke6 Nf6 h5 Ke2 Kf5 Kf3 Nd3 Bc3 Ba3 Nd7 Ke6 Ne5 Nxe5 Bxe5 Kf5 e4 Ke6 Bd4 Kf7 f5 Bc1 a4 h4 a5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c3 d5 Bg5 Ne4 Bf4 c6 Nd2 Bf5 Nxe4 Bxe4 f3 Bg6 h4 h6 e3 e6 h5 Bh7 Bd3 Bxd3 Qxd3 Nd7 Ne2 c5 e4 cxd4 exd5 e5 Qe4 Bd6 Bxe5 Nxe5 cxd4 f5 Qe3 Qa5 Kf1 Qxd5 f4 Bc7 Rc1 O-O-O fxe5 Kb8 Qb3 Qd7 Rh3 Bb6 Rc4 f4 Rh4 Rhf8 Qf3 Qb5 Qb3 Qe8 a4 g5 hxg6 Qxg6 Qf3 Qb1 Rc1 Qxb2 Rxf4 Rfe8 Qc3 Qxc3 Rxc3 Rxe5 Rf6 Rh5 Rc4 Rh1 Kf2 h5 Rf7 Rb1 g3 Rb2 Kf3 Rb3 Kg2 a5 Rh7 Rd5 Rh8 Ka7 Rcc8 Bxd4 Ra8 Kb6 Rh6 Kc7 Rh7 Kb6 Rh6 Kc7 Rh7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5 c6 dxc6 bxc6 Be2 h6 Nf3 e4 Ne5 Qc7 d4 exd3 Nxd3 Bd6 b3 O-O Bb2 Re8 h3 Ne4 O-O Qe7 Nc3 Nxc3 Bxc3 Qxe2 Bxa5 Bf5 Re1 Qxd1 Raxd1 Rxe1 Rxe1 Rb8 g4 Bg6 Kf1 h5 Re3 Rb5 Bd2 f6 f4 Rd5 Ke2 hxg4 hxg4 f5 c4 Rd4 Bc3 Re4 Rxe4 fxe4 Ne5 Bxe5 Bxe5 Be8 Ke3 Kf7 Kxe4 g6 Kd4 Ke7 Kc5 Kd7 Bb8 Kc8 Bxa7 Kb7 Bb6 Bd7 g5 Bf5 b4 Bc2 a4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Rb8 a3 a5 h4 Ne7 Nce3 Nxd5 Nxd5 Be6 g3 Qd7 Bg2 Bd8 O-O Bb6 Qd2 Rfc8 Rfd1 Bxd5 Qxd5 b4 axb4 axb4 Qxd6 Qg4 Qd3 bxc3 bxc3 g6 Rab1 Rd8 Qf3 Rxd1 Rxd1 Qxf3 Bxf3 Rc8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 Nbd7 Be3 Ng4 Bg5 f6 Bh4 h5 Nd2 Nh6 f3 Nf7 Qc2 Bh6 O-O-O c5 dxc6 bxc6 Kb1 a5 Na4 c5 Nc3 Be3 Nd5 Bd4 Nb3 Bb7 Nxd4 cxd4 f4 Rb8 Rhf1 Nh6 c5 Bxd5 exd5 Nf5 fxe5 Nxh4 exd6 Ne5 Rxd4 Nf5 Rxf5 gxf5 Qxf5 Kg7 Bxh5 Rh8 Rg4 Kf8 Qe6 Rb7 c6 Rxb2 Kxb2 Qb6 Ka3 Qc5 Ka4 Qc2 Kb5 Qb2 Ka6 Qe2 Kb7 Rh7 d7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 O-O Nc6 h3 Bh5 Be3 cxd4 cxd4 Be7 Nc3 Qd6 Nb5 Qb8 Ne5 Bxe2 Qxe2 O-O Nxc6 bxc6 Nc3 Nd5 Nxd5 cxd5 Rac1 Qb6 Rc2 Bf6 Qd3 a5 Rfc1 a4 Qd2 h6 Rc5 Rab8 R1c2 Qb4 Qd3 Qe1 Qf1 Qb4 Qd3 Qe1 Qf1 Qb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rad1 Bd7 Nf3 Nc6 Bf4 Qb7 Rfe1 b4 Ne2 e5 Bg5 Be6 Nh4 Na5 Bd5 Bxd5 Rxd5 Bxg5 Qxg5 Nc4 Ng3 Nxb2 Ngf5 f6 Qg3 Rc8 Qb3 Nc4 a3 g6 Ne3 Nxe3 Rxe3 Rf7 axb4 Qc6 c3 Nc7 Rdd3 Nb5 Nf3 Kg7 h4 h5 Nd2 Rfc7 Qa2 Qb7 Qa1 Rc6 Rg3 Qc7 Qd1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 d6 Bf4 e5 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Rb8 h4 Ne7 Nxf6 gxf6 Bd3 d5 exd5 Qxd5 Ne3 Qe6 Qh5 e4 Bc2 b4 c4 Kh8 O-O-O f5 Qg5 Rb6 h5 Rc6 Kb1 Rc5 h6 Qe5 Rh5 Rg8 Ng4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 Qxd5 e3 c5 a3 cxd4 axb4 dxc3 bxc3 b5 Nf3 O-O c4 bxc4 Bxc4 Qe4 Bd3 Qxb4 Ke2 Qb6 Ba3 Re8 Bc5 Qc7 Rhc1 a5 Bb4 Qxc2 Rxc2 Ba6 Rxa5 Bxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Ne2 Nf6 Nbc3 d6 g3 Nc6 Bg2 g6 d4 cxd4 Nxd4 Nxd4 Qxd4 Bg7 Be3 O-O Qd2 Ng4 Bf4 Ne5 Nd5 Bg4 O-O e6 Ne3 Nf3 Bxf3 Bxf3 Bxd6 b5 Qd3 Bxe4 Qxe4 Qxd6 Rfd1 Qc5 c3 Rab8 a3 a5 Rac1 h5 Rc2 Rfd8 Rcd2 Rxd2 Rxd2 b4 axb4 axb4 cxb4 Qxb4 Qxb4 Rxb4 Rd8 Bf8 Nd1 h4 Kg2 Kg7 Rd3 f5 b3 Kf6 Ne3 Bc5 f3 Bxe3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O Be6 f4 Qc7 f5 Bc4 Bg5 Nbd7 a4 O-O Kh1 Rfc8 a5 h6 Bxf6 Nxf6 Bxc4 Qxc4 Qf3 Rc7 Nd2 Qb4 Rfb1 Rac8 Ra4 Qc5 Rba1 Bd8 Nf1 b5 axb6 Qxb6 R1a2 a5 h3 Rc4 Rxc4 Rxc4 Nd2 Rb4 Ra1 Qc6 b3 a4 Rxa4 Rxa4 bxa4 Ba5 Nd5 Qxc2 Nxf6 gxf6 Nf1 d5 Ne3 Qc1 Kh2 dxe4 Qxe4 Bb6 Nd5 Bf2 Nxf6 Kh8 Qa8 Kg7 Nh5 Kh7 Nf6 Kg7 Nh5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nge2 a6 Ng3 Nbd7 a4 h5 Be2 e5 Be3 exd4 Bxd4 Ne5 h3 Qe7 f4 h4 Nf1 Nc6 Nd5 Qd8 Bc3 O-O Nd2 Nxd5 cxd5 Nd4 O-O Re8 Bd3 c6 f5 cxd5 f6 Bxf6 Nb3 dxe4 Rxf6 Qxf6 Bxd4 Qg5 Bc4 Bxh3 Qf1 Be6 Bxe6 Rxe6 Kh1 e3 Nc1 h3 Ne2 Re4 Qf3 hxg2 Kg1 Qh4
Result: 0-1
Processing game...
FEN: 
Moves: e3 d5 c4 dxc4 Bxc4 e5 d4 exd4 exd4 Bb4 Nc3 Nf6 Nf3 O-O O-O Bg4 h3 Bh5 g4 Bg6 Ne5 Nc6 Be3 Nxe5 dxe5 Nd7 f4 Nb6 Bb3 Bd3 Qf3 Bxf1 Rxf1 c6 f5 Qe7 f6 Qxe5 fxg7 Kxg7 Ne4 Nd5 Bxd5 cxd5 Ng3 Kg8 Nf5 Rac8 Qf2 Rc4 Nh6 Kh8 Bxa7 f6 Nf5 Re8 a3 Be1 Qg2 Re4 Nh6 Re7 Rf5 Re2 Rxe5 Rxg2 Kxg2 fxe5 Bb8 e4 Be5 Rxe5 Nf7 Kg7 Nxe5 Bd2 Kf1 Bc1 b3 Bxa3 g5 d4 Ke2 d3 Kd2 Bd6 Nc4 Bf4 Kc3 b5 Nb6 b4 Kb2 e3
Result: 0-1
Processing game...
FEN: 
Moves: e3 d5 c4 dxc4 Bxc4 e5 Nc3 Nc6 Nf3 f5 Qb3 Nh6 d4 exd4 exd4 Nxd4 Nxd4 Qxd4 O-O Be7 Bxh6 gxh6 Rfe1 c6 Ne2 Qh4 Rad1 Rf8 Nd4 Rh8 Be6 Bxe6 Qxe6 Rf8 Nxf5 Rxf5 Qxf5 Rd8 Rxd8 Kxd8 g3 Qf6 Qxh7 Bc5 Qd3 Kc7 Re2 Bd4 b3 Kd6 Kg2 c5 Qe4 Qf7 Qf4 Qxf4 gxf4 b5 f5 Be5 Kf3 c4 bxc4 bxc4 Ke4 Bf6 Rd2 Kc6 Rd4 Bxd4 Kxd4 Kd6 Kxc4 Ke5 Kb5 Kxf5 Ka6 Kg4 Kxa7 h5 Kb6 h4 a4 Kf3 a5 Kxf2 a6 Kg2 a7 Kxh2 a8=Q Kg3 Qg8
Result: 1-0
Processing game...
FEN: 
Moves: e3 d5 c4 dxc4 Bxc4 e5 d4 exd4 exd4 Bb4 Nc3 Nf6 Nf3 O-O O-O Bg4 Be3 a5 h3 Bh5 g4 Bg6 Ne5 Nbd7 f4 Nxe5 dxe5 Qe8 Qe1 Ne4 a3 Bxc3 bxc3 Qc6 Ba2 h6 f5 Bh7 Bd4 Ng5 Qe3 Rfe8 h4 Ne4 g5 hxg5 hxg5 g6 e6 fxe6 fxe6 Re7 Rae1 b5 Qxe4 Qxe4 Rxe4 b4 Rf7 b3 Bxb3 c5 Rxe7 cxd4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 exd5 cxd5 c4 Nf6 cxd5 Nxd5 Nc3 e6 Nf3 Be7 Bc4 Nxc3 bxc3 Qc7 Qd3 O-O O-O Nd7 Bg5 Bxg5 Nxg5 Nf6 Rae1 Bd7 Bb3 h6 Ne4 Nxe4 Rxe4 Rac8 Re3 Kh8 Bc2 f5 Bb3 a5 Rfe1 a4 Bc2 b5 Rb1 Qd6 Qd2 Rc7 Rbe1 Rfc8 Bd1 b4 cxb4 Rc4 a3 Rxd4 Qe2 Rd2 Qf1 f4 Re4 Qd5 Bg4 Rcc2 Rxf4 Qg5 Rf8 Kh7 Bxe6 Bxe6 Rxe6 Qd5 Ree8 Rd1 Rh8 Kg6 Re6 Kf7 Re1 Rcc1 Rc8 Rxe1 Rxc1 Rxf1 Rxf1 Qb3 b5 Qxb5 h3 Qb2 Rd1 Qxa3 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 c4 g6 d4 Bg7 Nc3 d6 Nf3 Nd7 Be2 e5 O-O Ngf6 dxe5 dxe5 Qd6 Qe7 Qxe7 Kxe7 b3 Ke8 Bb2 Ng8 Rad1 f6 Rd6 Bf8 Rd3 Nh6 Rfd1 Nf7 Nd2 Nc5 Rf3 Be7 Rg3 Ne6 Bd3 Nf4 Nf1 h5 Ne2 h4 Re3 Ne6 Bc2 a5 h3 b6 Red3 Nc5 R3d2 Ng5 Nc3 Nge6 Ne3 Nd4 Bb1 Nce6 Nc2 Nxc2 Bxc2 Kf7 Na4 Rb8 Kh1 Nc5 Nxc5 Bxc5 Bb1 Be6 f3 Ke7 Rf1 g5 Rfd1 Rhd8 Bd3 a4 Be2 Rxd2 Rxd2 axb3 axb3 Bb4 Rd1 Ra8 Ra1 Rxa1 Bxa1 Bc5 Bb2 Kd6 Bd3 Bd4 Ba3 c5 Bb1 b5 cxb5 Bxb3 Bd3 Kc7 Be2 Kb6 Kh2 c4 Be7 Be3 Bxf6 Bf4 Kg1 c3 Bd3 c2 Bd8 Kb7 Bxc2 Bxc2 Kf2 Bd3 b6 Bc4 Bc7 Bg3 Kg1 Kc6 Bd8 Bf4 Kf2 Ba6 Bc7 Kb5 Bd8 Kc4 Ke2 Kc3 Kd1 Kd3 Ke1 Kc2 Kf2 Kd2 Kg1 Ke1 Bc7 Ke2 Bb8 Be3 Kh2 Bd4 Bc7 Kf2 Bd8 Be3 b7 Bxb7 Bc7 Ba6 Ba5 Bf1 Bb6 Bxb6 f4 exf4 e5 Bxg2 e6 Kf1 e7 Bg1
Result: 0-1
Processing game...
FEN: 
Moves: e4 g6 d4 Bg7 Nc3 c6 Bc4 d6 Nf3 Nf6 O-O O-O Bg5 Nxe4 Nxe4 d5 Bd3 dxe4 Bxe4 Nd7 c3 Nf6 Bxf6 Bxf6 Re1 Be6 Bd3 Qd6 Nd2 Bg7 Qe2 Rae8 Ne4 Qc7 Nc5 Bc8 Bc2 b6 Nd3 Bb7 b4 e6 a4 Rd8 a5 Qc8 axb6 axb6 Ra7 Qc7 Rea1 Ra8 Rxa8 Rxa8 Rxa8 Bxa8 f4 c5 bxc5 bxc5 Nxc5 Qxf4 g3 Qc1 Kf2 Qh1 Ke3 Bh6 Kd3 Bf3 Qf2 Bd5 c4 Bc6 Kc3 Qa1 Kd3 Qa3 Ke2 Qc3 Bb3 Bg7 Qe3 Qxd4 Nd3 Qg4 Kd2 Bd4 Qf4 Qxf4 gxf4 Kg7 Nb4 Bb7 Nc2 Bc5 Ne1 Bb4 Ke2 Bd6 Nd3 Be4 Nf2 Bf5 Kf3 h5 Bd1 Bc7 c5 Kf6 Ne4 Ke7 Ba4 Bg4 Kg3 e5 fxe5 Bxe5 Kg2 Be6 Ng3 Bd5 Kh3 f5 Bc2 Kf6 Ba4 f4 Nf1 f3 Kh4 Bd4 c6 Bb6 Bb5 Bc7 Ne3 Bf7 Bc4 g5 Kh3 g4 Kh4 Bxc4 Nxc4 f2 Nd2 Bxh2 Kxh5 Bf4 Ne4 Ke6 Nxf2 g3 Nh3 Bd6 Kg4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 g6 c4 Bg7 Nf3 d6 g3 c5 d5 b5 cxb5 a6 bxa6 Qa5 Nc3 Nf6 Bd2 O-O Bg2 Qxa6 O-O Nbd7 Re1 Nb6 e4 Bg4 h3 Bxf3 Bxf3 Nfd7 Be2 c4 Be3 Rfc8 a4 Qa5 Bg5 Kf8 Qd2 Nc5 Ra3 Qb4 Bh6 Bxh6 Qxh6 Kg8 Rea1 Ncxa4 Nxa4 Rxa4 Qe3 Rxa3 Qxa3 Qd2 Qa5 Qxb2 Qe1 c3 Bd3 Qb4 Rb1 Qd4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 O-O Bd7 Nd2 Nc5 b3 Nfxe4 Bxd8 Nxc3 Qe1 Rfxd8 Rc1 Nxa2 Ra1 Nb4 Bd1 e4 Rb1 Re8 Qe3 f5 h4 Rf8 g3 Rae8 Kg2 Nbd3 Rg1 f4 gxf4 Rxf4 h5 g5 Rf1 Rh4 Rh1 Rf4 Rf1 Ref8 f3 Rh4 fxe4 Nf4 Kg1 Ncd3 e5 Nxe5 Rc1 Rh3 Nf3 g4 Nxe5 Rxe3 Nxd7 Nh3 Kg2 Rxf1 Kxf1 g3 Kg2 Nf4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 d5 e3 O-O Be2 Bg4 O-O c6 c4 Bxf3 Nxf3 Nbd7 Bh4 Re8 Qc2 Nb6 b3 Rc8 Rac1 Ne4 Bd3 Nf6 Rfd1 Qd7 c5 Na8 b4 Nc7 a4 a6 Ne5 Qe6 Re1 Nd7 Nf3 Nf6 h3 Qd7 Bg3 Nh5 Bh2 f5 Ne5 Qe6 Rb1 Nf6 Rec1 Nd7 Nf3 Nf6 b5 axb5 axb5 Ne4 b6 Nb5 Bxb5 cxb5 Rxb5 Qc6 Ra5 Ra8 Rca1 Rxa5 Rxa5 Rc8 Qa4 Qe6 Ra8 Bf6 Be5 Kf7 Rxc8 Qxc8 Bxf6 exf6 Qb3 Qc6 Ne1 Nd2 Qa2 Nc4 Nd3 Qb5 Nf4 Qc6 h4 g5 hxg5 fxg5 Nd3 Qb5 c6 Qxc6 Qa8 Qxb6 Nc5 Qc7 Qh8 Kg6 Qg8 Kh6 Qe6 Kg7 Qxf5 Qe7 Qxd5 Nd6 Ne6 Kg6 Nxg5 Qxg5 Qxd6 Kh5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7 Qg4 O-O Bd3 Nbc6 Qh5 Ng6 Nf3 Qc7 Be3 c4 Bxg6 fxg6 Qg4 Qf7 Ng5 Qf5 Qe2 Nxe5 dxe5 Qxe5 Qd2 Bd7 O-O h6 f4 Qd6 Nf3 Rf5 Rae1 Raf8 Nh4 R5f6 Bd4 Rxf4 Rxf4 Qxf4 Qxf4 Rxf4 Nxg6 Re4 Rxe4 dxe4 Ne5 Ba4 Nxc4 Bxc2 Bxa7 Bd3 Ne5 Bb5 c4 Be8 Kf2 g5 Ke3 Kg7 Kxe4 Kf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 h3 O-O Bg5 c6 Bd3 e5 d5 Na6 a3 Nc5 Bc2 cxd5 Nxd5 Ne6 Be3 Nxd5 cxd5 Nf4 Qf3 f5 exf5 Bxf5 Bxf5 Rxf5 Bxf4 Rxf4 Qb3 Qa5 Kf1 Raf8 f3 e4 Re1 Bd4 Ne2 Qa6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 g6 b3 Bg7 Bb2 c5 c4 d6 Bg2 e5 O-O Nc6 Nc3 O-O d3 Ne8 Ne1 Nc7 Nc2 Bd7 Ne3 Rb8 Ned5 Nxd5 Nxd5 b5 Bc3 Be6 Qd2 Qd7 e3 Ne7 Nxe7 Qxe7 f4 Rfd8 Rac1 d5 cxd5 Bxd5 Bxd5 Rxd5 fxe5 Bxe5 Bxe5 Rxe5 e4 Rd8 Qe3 f5 Rf4 Qd6 Qf2 fxe4 dxe4 Qd4 Re1 Qxf2 Kxf2 Rd2 Re2 Rd4 h4 c4 bxc4 Rxc4 g4 b4 g5 Re7 Kf3 Rf7 Rd2 a5 Rf6 Rxf6 gxf6 Kf7 Rd6 a4 e5 b3 axb3 axb3 Rd7 Ke6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2 cxd4 cxd4 Qa5 Qd2 Qxd2 Bxd2 b6 d5 Na6 Be3 f5 e5 f4 Bd4 Bf5 Rc1 Nb4 Bc4 Nd3 Kd2 Nxc1 Rxc1 Rfc8 d6 Kf8 Ng5 exd6 Nxh7 Ke8 Bb5 Kd8 Rd1 dxe5 Bc3 Rxc3 Kxc3 Ke7 Ng5 Rc8 Kb3 Bc2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 cxb5 a6 b6 d6 Nc3 Qxb6 e4 g6 Nf3 Bg7 a4 O-O a5 Qc7 Bc4 Bg4 h3 Bxf3 Qxf3 Nbd7 Qe2 Ne5 Bxa6 c4 Bb5 Nd3 Kf1 Rxa5 Rb1 Nd7 Bxd7 Qxd7 g3 Rb8 Bd2 Ra7 Kg2 Rab7 b4 Rxb4 Rxb4 Rxb4 Ra1 h5 Ra8 Kh7 Nd1 Rb1 Ne3 Qb5 Qf3 Ne5 Qf4 Qb2 Nf5 c3 Bxc3 Qxc3 Nxe7 Qe1
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Qb6 Be3 Nc6 Na4 Qa5 c3 c4 b4 Qc7 g4 b5 Nc5 a5 a3 Nxc5 bxc5 Be7 Bh3 h5 gxh5 Rxh5 Bg4 Rh8 h4 Rb8 Bd2 b4 axb4 axb4 cxb4 Nxb4 Bxb4 Rxb4 Rb1 Qb7 Rxb4 Qxb4 Kf2 Bd7 h5 Ba4 Qa1 c3 Rb1 Qc4 Rb8 Bd8 Qb1
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O O-O Qe2 Bg6 Ne5 Nbd7 Nxg6 hxg6 Rd1 Qe7 e4 e5 d5 Nb6 Bb3 a5 Be3 Bc5 d6 Bxd6 Bxb6 Nd7 Be3 Nc5 Bc4 Rad8 Rd2 Qc7 Rad1 Be7 g3 b6 h4 Rxd2 Rxd2 Rd8 Qf3 Bf6 Rxd8 Qxd8 Bxc5 bxc5 Qd3 Qc7 b3 g5 h5 Kh7 Qf3 Kh6 Nd1 g4 Qxg4 Bg5 Nc3 Qe7 Kg2 Bd2 Ne2 Qf6 Ng1 Bc3 Nf3 Bd4 Qc8 g6 Qf8 Qg7 Qe7 gxh5 Bxf7 c4 Qe6 Kh7 Qf5 Kh8 bxc4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O e4 d6 Be2 e5 O-O Nc6 d5 Ne7 b4 Nh5 c5 Nf4 Bxf4 exf4 Rc1 a5 cxd6 cxd6 Nb5 Bg4 Rc7 axb4 Qd2 Bxf3 Bxf3 Be5 Rxb7 Qa5 Nd4 Qxa2 Qxa2 Rxa2 Rxb4 Rfa8 Bg4 Bxd4 Rxd4 g5 h4 gxh4 Kh2 Ng6 Kh3 Rb2 Bf5 Ne5 Kxh4 h6 Bh3 Kg7 Rdd1 Raa2 f3 Ng6 Kg4 Kf6 Rb1 h5 Kxh5 Ra8 Kg4 Rh8 g3 Rh2 Rh1 R8xh3 Rxh2 Rxh2 gxf4 Rg2 Kh3 Nxf4 Kh4 Ke5 Rb7 Kd4 Rxf7 Ke3
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O a6 Bd3 Nc6 Nc3 Qc7 a3 b6 dxc5 bxc5 Ne4 Be7 Bd2 Bb7 Bc3 Nxe4 Bxe4 O-O Rc1 h6 Qa4 f5 Bb1 e5 Nh4 Bxh4 Qxh4 Qf7 f3 Rac8 Rcd1 Qe6 Bd3 f4 Rfe1 Ne7 exf4 Ng6 Qg3 Nxf4 Bxe5 Nh5 Qh4 c4 Bf1 Qg6 Rd6 Qf7 Qg4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 f4 e6 Nf3 Nge7 g3 d5 d3 g6 Bg2 Bg7 O-O b6 e5 h5 Ne2 Nf5 c3 a5 Rf2 Ba6 h3 Bf8 Kh2 Be7 Neg1 Rc8 Re2 d4 c4 b5 b3 Kf8 Ng5 Bxg5 fxg5 Qb6 Bf4 Bb7 Be4 h4 g4 Ng3 Bxg3 hxg3 Kxg3 Qc7 Bxc6 Bxc6 Qf1 Ke8 Qf6 Rh7 h4 Kd7 Rf1 Qb6 Qf4 bxc4 bxc4 Qb4 Qd2 Rch8 Qxb4 axb4 Rh2 Ra8 Nh3 Ra3 Nf4 Ke7 Re1 Rg7 Rb2 Rg8 Ree2 Ra7 Rh2 Raa8 Rbd2 Ra3 Nh3 Be4 Nf2 Bc6 Kf4 Rga8 h5 Rxa2 Rxa2 Rxa2 h6 Ra8 Nd1 Kf8 Nb2 Kg8 Kg3 Ra1 Rf2 Be8 Re2 Rg1 Kh4 Bc6 Rf2 Re1 h7 Kh8 Rxf7 Re3 Nd1 Rxd3 Nf2 Rf3 Rf6 b3 Nd1 Rd3 Nb2 Re3 Nd1 Rxe5 Rxg6 Kxh7 Rf6 Kg7 Rf2 Re1
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 c5 g3 g6 c3 Bg7 d4 cxd4 cxd4 d5 Bg2 Nc6 Nc3 e6 O-O Nge7 b3 O-O e3 Bd7 Qd2 Qa5 Na4 b6 Qxa5 Nxa5 Ba3 Rfe8 Nc3 Rac8 Rfc1 Rc6 Bf1 Rec8 Bb5 Rxc3 Bxd7 Rxc1 Rxc1 Rxc1 Bxc1 Nec6 Bd2 Bf8 Bxa5 Nxa5 Ne5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5 e4 c6 b3 Re8 h3 a6 Be3 exd4 Nxd4 Nc5 Qc2 Qe7 Rfe1 Bd7 f3 Nh5 g4 Nf4 Bxf4 Bxd4 Kh1 Ne6 Bh2 Qf6 Rac1 Be5 Ne2 Bxh2 Kxh2 Qe5 Kh1 h5 Rcd1 hxg4 hxg4 Rad8 Qd2 Bc8 Qh6 g5 Kg1 b5 cxb5 axb5 Rc1 Bb7 Kf2 Ra8 Rc2 Nd4 Nxd4 Qxd4 Kg3 Qe5 Kf2 Re6 Qh5 Kf8 Qh7 Rf6 Rd1 Ke7 Kg1 Rf4 Qh2 Rh8 Qg3 Rxe4 Qf2 Re3 Qd2 c5 Qa5 Kf6 Qc7 Re1
Result: 0-1
Processing game...
FEN: 
Moves: d4 f5 g3 Nf6 Bg2 e6 c4 c6 Nd2 d5 Qc2 Bd6 Nh3 O-O Nf3 Bd7 O-O Be8 Bf4 h6 Bxd6 Qxd6 Nf4 Nbd7 Nd3 dxc4 Qxc4 Bh5 b4 Ne4 Rfd1 Nb6 Qb3 Nd5 Rac1 f4 Nc5 Nxc5 bxc5 Qe7 Rd2 Bxf3 Bxf3 Rad8 Qd3 Qf6 Rb1 Rd7 Rb3 Ne7 Qe4 fxg3 hxg3 Nf5 Rbd3 Rfd8 e3 g6 Rb3 Rf8 Bg2 h5 Rdb2 Rff7 Bh3 g5 Bxf5 exf5 Qe5 Kg7 a4 a6 a5 Rfe7 Qxf6 Kxf6 Kf1 Ke6 Ke2 f4 gxf4 gxf4 Kd3 Kf5 Rb1 h4 Rh1 Kg4 Rbb1 Rh7 f3 Kxf3 Rbf1 Kg4 Rxf4 Kg3 Ke2 h3 Rg1 Kh2 Rg6 Rdf7 Rxf7 Rxf7 e4 Rf4 Ke3 Rf1 e5 Rg1 Rxg1 Kxg1 e6 h2 e7 h1=Q e8=Q Qh3 Kd2 Qg2 Qe2 Qd5 Qg4 Kf2 Qf4 Qf3 Qxf3 Kxf3 Kd3 Kf4 d5 cxd5 Kd4 Kf5 Kxd5 Kf6 Kd6 Kf5 Kc7 Ke5 Kxb7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 c4 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 Bb4 Nxc6 bxc6 Bd3 e5 O-O O-O Be3 d6 Na4 d5 exd5 cxd5 a3 Be7 cxd5 Nxd5 Bc5 Nf6 Re1 Bxc5 Nxc5 Qb6 Qc2 Re8 h3 a5 Rac1 Rb8 Re2 h6 Bc4 Re7 Rd2 e4 Na4 Qa7 Rd8 Kh7 Ba2 Bd7 Rxb8 Qxb8 Nc5 Bf5 Qb3 Qd6 Qe3 Bg6 b4 axb4 axb4 Qb8 Rb1 Ra7 Bc4 Qe5 Nd3 Qc7 Nf4 Rb7 Bf1 Bf5 b5 g5 Ne2 Nd5 Qd4 e3 Rc1 exf2 Qxf2 Qe5 Rc5 Be6 Nd4 Qf4 Nxe6 Qxf2 Kxf2 fxe6 Bc4 Nc7 Ke3 Kg7 Kd4 Kf6 g4 Ke7 Ke5 Kd7 Kf6 Rb6 Kg7 Kd6 Rc6 Rxc6 bxc6 Kxc6 Kxh6 Kd6 Kxg5 Ke7 Kg6 Ne8 h4 Nd6 Bb3 Nf7 g5 Ne5 Kh6 Nf3 Kh5 Kf7 Kg4 Ne5 Kf4 Ng6 Kg3 Ne7 h5 Nf5 Kf4 Nd4 Bc4 Nf5 Ke5 Ng3 Bxe6 Kg7 Bg4 Nf1 h6 Kh7 Bf5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 dxc4 e4 b5 e5 h6 Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7 exf6 Bb7 g3 c5 d5 Qb6 Bg2 O-O-O O-O b4 Rb1 Qa6 dxe6 Bxg2 e7 Bxf1 Qd5 Bxe7 fxe7 Rdg8 Ne4 Rg6 Rxf1 Qc6 Qxc6 Rxc6 Rd1 Re8 Nd6 Rxd6 Rxd6 f6 Bxf6 Kc7 Re6 Nxf6 Rxf6 Rxe7 Kf1 Re4 Rf4 Rxf4 gxf4 Kd6 Ke2 a5 a4 c3 bxc3 b3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e5 Nc3 Bb4 Nd5 Be7 d4 d6 e4 c6 Nxe7 Qxe7 Ne2 f5 dxe5 Qxe5 exf5 Nf6 Qd4 Bxf5 Bf4 Qa5 Qc3 Qxc3 Nxc3 O-O O-O-O d5 Bd6 Rc8 f3 Nbd7 g4 Be6 g5 Ne8 cxd5 Bxd5 Bg3 Be6 Bd3 Nc5 Bc2 Rd8 h4 Bf7 Ne4 Nxe4 fxe4 Kf8 Rdf1 Kg8 h5 Rd7 Rf2 Nd6 g6 Be6 gxh7 Kh8 h6 g6 Rf6 Re8 Rxg6 Nc4 Be1 Kxh7 Rg3 Ne5 Bc3 Bc4 Rg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Bg4 Be3 Nfd7 Qb3 c5 d5 Na6 Be2 Rb8 Bf4 Bxf3 Bxf3 Ne5 Be2 b5 O-O c4 Qa3 Rb6 Nxb5 Rxb5 Qxa6 Rxb2 Rae1 Qd6 Qxd6 exd6 Bxe5 Bxe5 Bxc4 Rc8 Bb3 Bd4 g3 Re8 Kg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Bd3 Nc6 Be3 Nf6 O-O Ne5 h3 Bc5 Kh1 d6 f4 Ned7 a3 O-O Qe1 Qb6 Qf2 e5 fxe5 dxe5 Nf5 Bxe3 Nxe3 Qxb2 Ned5 Nxd5 Nxd5 b5 Qg3 Nc5 Rab1 Qxa3 Ra1 Qb2 Rab1 Qa3 Ra1 Qb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Rb8 h4 Ne7 Nxf6 gxf6 Qd2 Bb7 Bd3 d5 exd5 Qxd5 O-O-O e4 Be2 Qxa2 Qh6 Qe6 Nd4 Qb6 Rh3 Kh8 Bg4 Rg8 Ne6 Rg6 Qf4 Re8 Rd6 Nd5 h5 Nxf4 hxg6 Qxd6 Rxh7 Kg8 gxf7 Kxh7 fxe8=Q Nxe6 Bf5 Kg7 Qg6 Kf8 Qxf6 Ke8 Bxe6 Qf8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Kh1 b5 Nd5 Nxd5 Qxd5 Ra7 Be3 Be6 Qd1 Rb7 Rc1 Nc6 c4 Na5 Nxa5 Qxa5 cxb5 axb5 a3 Qa8 Bd3 Rd8 f3 h6 Rf2 Bg5 Bxg5 hxg5 Rd2 Qa5 h3 g6 Kh2 d5 b4 Qxa3 exd5 Bd7 d6 Qxb4 Be4 Rb6 Rc7 Kg7 Rd3 Be6 d7 Qe7 Bc6 b4 Qa1 Bf5 Rd5 b3 Bb5 Qf6 Qc3 e4 Qxb3 Be6 Rc8 Rxd7 Qa3 Rxd5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3 d6 g4 a6 Be3 Nge7 Nb3 b5 f4 Bb7 Qf3 Na5 O-O-O Nxb3 axb3 Rc8 h4 Nc6 g5 Qa5 Kb1 Nb4 Bd4 d5 f5 dxe4 Qh3 Bd6 Qe3 exf5 Bh3 O-O Bxf5 Rxc3 Bxc3 Qa2 Kc1 Nd5 Rxd5 Qa1 Kd2 Qxh1 Rxd6 Qh2 Kc1 Qxd6 b4 Rd8 Bg4 Bc8 Be2 Be6 b3 Qh2 h5 Qc7 h6 Rc8 Kb2 f6 gxf6 gxf6 Qd4 Qe5 Qg1 Qg5 Qb6 Qd5 Bg4 Bxg4 Qxf6 Qd7 Qh8 Kf7 Qxh7 Ke6 Qxe4 Kf7 h7 Bf5 Qf4 Qe6 h8=Q Rxh8 Bxh8 Qe4 Qc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7 Qe2 Bg4 h3 Bxf3 Qxf3 O-O Rd1 Qa5 e4 e5 d5 Nb6 Bb3 Bxc3 bxc3 cxd5 exd5 Rac8 c4 Rfe8 Bd2 Qa6 d6 Nbd7 Be3 Rc6 a5 Rxd6 Ba4 Rxd1 Rxd1 Rd8 c5 h6 Qf5 g6 Qc2 Qxa5 Rd6 Kg7 Qd1 Qc7 h4 Kh8 h5 g5 Qf3 Kg7 Qf5 b6 Bxd7 Nxd7 Bxg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 h3 O-O Bg5 c6 Bd3 e5 d5 Na6 Nge2 Nc5 Bc2 Qb6 O-O cxd5 cxd5 Bd7 Rb1 a5 Qd2 Rfc8 Kh1 Qd8 a3 a4 Rbe1 Ra6 Nc1 Qa5 Bb1 Nh5 Nd3 Nb3 Qe3 f6 Bh6 g5 Bxg7 Kxg7 Kh2 Nf4 h4 h6 Nb4 Raa8 g3 Ng6 f4 Qc5 f5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e5 Nf3 Bb4 Bd2 Bxd2 Nbxd2 exd4 Bxc4 Nc6 O-O Qf6 b4 a6 e5 Qg6 Nb3 Nge7 Nbxd4 Bh3 Nh4 Qg4 Qxg4 Bxg4 h3 Nxe5 hxg4 Nxc4 Rfc1 b5 a4 Kd7 axb5 axb5 Rxa8 Rxa8 Nxb5 c6 Rxc4 cxb5 Rc5 Rb8 Nf3 Nc6 Ne5 Nxe5 Rxe5 Kc6 g5 Rd8 Rc5 Kb6 Rf5 Rd7 f4 Kc6 Rc5 Kb6 Rf5 Kc6 Kf2 g6 Rc5 Kb6 Ke3 Rd6 g4 Rd1 Rc8 Rb1 Rf8 Rxb4 Rxf7 Rc4 Rxh7 b4 Rh1 b3 Rf1 b2 Rb1 Rb4 f5 Kc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 c5 d4 Bg7 Rb1 O-O Be2 cxd4 cxd4 Qa5 Bd2 Qxa2 O-O Bg4 Bg5 h6 Bxe7 Re8 Rxb7 Nc6 Bc5 Rxe4 Bd3 Rxd4 Bxd4 Nxd4 Be4 Rd8 Qb1 Ne2 Kh1 Qc4 Bxg6 Nc3 Bxf7 Qxf7 Rxf7 Nxb1 Rxa7 Nc3 h3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 f3 Bg7 e4 d6 Nc3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 d5 cxd5 cxd5 a6 Ng3 h5 Bd3 b5 O-O h4 Nge2 Nh5 a4 b4 Na2 Nc5 Bc4 a5 Bxc5 dxc5 Nac1 Kh7 Nb3 Bh6 Qd3 Qd6 Bb5 c4 Bxc4 Kg7 Bb5 Qb6 Kh1 Rh8 d6 Be6 Bc4 Bxc4 Qxc4 Qxd6 Rad1 Qe7 Qb5 Be3 Rd7 Ng3 Nxg3 hxg3 h3 Qe6
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 d6 g3 Bg4 Bg2 c6 Nc3 Nbd7 d3 e5 h3 Bh5 Nh4 Be7 Nf5 O-O O-O Re8 g4 Bg6 Nxe7 Qxe7 f4 exf4 Bxf4 Nc5 Bg5 Ne6 Bxf6 gxf6 Qd2 Nd4 Rae1 Kg7 e3 Ne6 Ne4 Bxe4 Bxe4 Rh8 Rf5 h5 gxh5 Ng5 h4 Nxe4 Qg2 Kf8 Qxe4 Qxe4 dxe4 Ke7 Ref1 Rh6 Kg2 Rah8 Kg3 Rxh5 Rxh5 Rxh5 Rf5 Rh8 h5 Rg8 Kf2 Rg4 Rf4 Rg5 Rf5 Rg4 Rf4 Rg7 h6 Rh7 Rh4 f5 exf5 Kf6 Rh5 Ke5 Kf3 f6 e4 Kd4 b3 a6 Rh2 b5 cxb5 axb5 a4 bxa4 bxa4 Kc5 Kg4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O h3 Na6 Bg5 c6 Be2 e5 d5 h6 Be3 cxd5 cxd5 Nh5 Nd2 Nf4 Bf1 Nc5 Nb3 b6 g3 Nh5 Qd2 Kh7 Be2 Nf6 Nxc5 bxc5 g4 Rb8 f3 Bd7 h4 h5 Bg5 Qa5 Rg1 Rb4 Bxf6 Bxf6 gxh5 Bxh4 Kf1 Bh3 Rg2 gxh5 Kg1 Bxg2 Kxg2 Rg8 Kh1 Bg5 f4 Bxf4 Qc2 Qd8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 g3 b5 Bg2 Bb7 Bg5 c5 c3 Qb6 Bxf6 gxf6 O-O Na6 a4 b4 a5 Qc7 e3 Rb8 Nh4 Bxg2 Nxg2 Qc6 Nf4 Nc7 Nd2 bxc3 bxc3 Bd6 Nh5 Be7 Qg4 Nd5 c4 Nc3 Rfc1 Rb2 Rxc3 Rxd2 Rb3 Qc7 Rab1
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O O-O Qe2 Bg6 Ne5 Nbd7 Nxg6 hxg6 Rd1 Qa5 h3 Rad8 Bd2 e5 Be1 exd4 Rxd4 Nb6 Bb3 Rxd4 exd4 Re8 Qd3 Nbd5 Bd2 Qb6 a5 Qc7 Nxd5 Nxd5 Bxb4 Nxb4 Qxg6 Nd5 Qf5 g6 Qf3 Qd6 a6 b6 g3 Kg7 Rc1 Re7 Kg2 Nb4 Qc3 Kg8 Re1 Rxe1 Qxe1 Kf8 Qc3 Nxa6 Qc4 Qd5 Qxd5 cxd5 Bxd5 Nb4 Bc4 Nc6 d5 Ne5 Bb5 f6 f4 Nf7 Kf3 Ke7 h4 f5 Ke3 Nd6 Be2 Ne4 g4 Nf6 gxf5 Nxd5 Ke4 Nf6 Ke5 Nd7 Kd5 gxf5 h5 Kf6 Kc6 Nc5 Bc4 Ne4 Bd5 Ng3 Bf3 Ne4 Kb7 Nd6 Kxa7 Nc4 b4 Kg7 Be2 Ne3 Kxb6 Nd5 Kc5 Nxf4 Bf3 Kf6 b5 Ne6 Kc4 Nd8 b6 Kg5 Kd5 Nb7 Kc6 Nd8 Kd7 Nf7 b7 Ne5 Kd6 Nxf3 b8=Q Kxh5 Qf8 Kg4 Qg7 Kf4 Kd5 Ke3 Qg3 f4 Qh3 Ke2 Qg4 Ke3 Qg6 Nh4 Qe4 Kf2 Qxf4 Ke2 Qxh4 Kd3 Qg3 Ke2 Ke4 Kd2 Kd4 Ke2 Qg2 Kd1 Kd3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O O-O Nh4 Nbd7 h3 Rc8 Qe2 Nb6 Bb3 Nbd5 Bd2 Qe7 Rfc1 Rcd8 Qf3 Ne4 Nxf5 exf5 Nxe4 fxe4 Bxb4 Nxb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 h3 Bh5 Be3 cxd4 cxd4 Bb4 Nc3 O-O O-O Qa5 Qb3 Nc6 a3 Bxc3 bxc3 Nd5 Rac1 Qc7 c4 Nxe3 Qxe3 Rfd8 Rfd1 Ne7 g4 Bg6 Ne5 Rac8 d5 b6 h4 f6 Nxg6 Nxg6 dxe6 Qe5 Qxe5 fxe5 Rxd8 Rxd8 c5 Rc8 Rd1 Kf8 cxb6 axb6 Rd7 Nf4 Bd3 Nxe6 Bxh7 Nf4 Be4 Rc1 Kh2 Rc3 f3 Rxa3 h5 Ne6 Rd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7 Qg4 Kf8 h4 Qc7 Qd1 cxd4 cxd4 Qc3 Bd2 Qxd4 Nf3 Qe4 Be2 Nbc6 h5 Nxe5 h6 gxh6 Bxh6 Kg8 Rb1 N7g6 Rb4 Nxf3 gxf3 Qe5 f4 Qc3 Kf1 f5 Rb3 Qf6 c4 b6 cxd5 Bb7 Rd3 Re8 Rg1 b5 dxe6 Rxe6 Rd8 Kf7 Rd7 Re7 Rxb7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O c5 d5 b5 cxb5 a6 bxa6 Bf5 Nc3 Ne4 Nxe4 Bxe4 Nd2 Bxg2 Kxg2 Rxa6 Nc4 Nd7 e4 Qa8 a4 Rb8 Ra3 Nb6 Nxb6 Raxb6 b3 Rb4 Re1 Qb7 a5 Qb5 Re2 Rxb3 Rxb3 Qxb3 Qxb3 Rxb3 Ra2 Rb8 Kf3 Kf8 Ke2 Ke8 Bd2 Bd4 Ra4 Kd7 a6 Ra8 f3 h5 h3 Kc8 Kd3 Kb8 Kc4 Ka7 Kb5 Rb8 Kc6 Rb3 g4 Rxf3 gxh5 gxh5 Ba5 Rxh3 Bb6 Ka8 Bd8 Be5 Bxe7 Rg3 Ra5 Ka7 Rb5 Rg8 Rb3 c4 Ra3 c3 Ra2 c2
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 f4 O-O Nf3 Na6 Be2 e5 fxe5 dxe5 d5 Nc5 Bg5 h6 Bxf6 Qxf6 b4 Na6 a3 c5 Rb1 Bd7 b5 Nc7 d6 Ne6 Nd5 Qd8 Qd2 Nd4 O-O Be6 Nxd4 Bxd5 Nc6 bxc6 cxd5 cxb5 Rxb5 Qxd6 Qa5 f5 Rxc5 fxe4 Rc6 Rxf1 Bxf1 Qf8 d6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bb6 a4 a5 b5 Nd4 Nxd4 Bxd4 c3 Bb6 d4 exd4 O-O Ne7 Bg5 h6 Bxe7 Qxe7 cxd4 Qd6 Nc3 Bxd4 Nd5 Bxa1 Qxa1 O-O e5 Qc5 Rc1 c6 Ba2 Qa3 Nb6 d5 Nxa8 Kh8 Nb6 Be6 h3 Rd8 bxc6 bxc6 Rc3 Qb4 Rxc6 Rb8 Nxd5 Qxa4 Rc1 Qa3 Bc4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3 d6 Be3 Nf6 f3 Be7 g4 O-O Qd2 a6 O-O-O Nxd4 Bxd4 b5 Kb1 Bb7 h4 Rc8 g5 Nd7 Rg1 b4 Ne2 Ne5 Rg3 Nc4 Qc1 e5 Bf2 a5 Bg2 Ba6 Re1 a4 Bh3 Rc6 Qd1 d5 exd5 Rd6 f4 Rxd5 Rd3 Na3 bxa3 Bxd3 cxd3 Rxd3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Bd3 Nf6 O-O Ne5 h3 Bc5 Kh1 d6 f4 Ned7 a3 b5 Bxb5 axb5 Ndxb5 Qb6 Bxc5 dxc5 Nd6 Ke7 Nxc8 Rhxc8 e5 Ne8 Qh5 h6 Rae1 f5 Rf3 c4 g4 fxg4 Qxg4 Ra5 Ne4 Qc6 Nd6 Nxd6 exd6 Kf8 Rg1 g5 Rg3 Rf5 Qh5 Nf6 Qxh6 Kf7 Kg1 Rg8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 Nc6 Qd2 O-O O-O-O d5 exd5 Nxd5 Nxc6 bxc6 Bd4 Nxc3 Qxc3 Bh6 Be3 Bxe3 Qxe3 Qb6 Qxe7 Be6 Qa3 Rfd8 Ba6 c5 Be2 c4 f4 Rd4 Rxd4 Qxd4 g3 c3 Qxc3 Qxc3 bxc3 Bxa2 Kb2 Be6 c4 Kf8 Ra1 a5 c5 Ke7 c4 Bd7 Bf3 Rb8 Kc3 a4 Ra3 Rc8 Kd4 h5 Bb7 Rd8 Kc3 h4 gxh4 Rh8 Bd5 Rxh4 Kb4 Rxh2 c6 Be6 Bxe6 fxe6 Kc5 Kd8 Rxa4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Re8 Rd1 Bf5 d6 h6 Bf4 Nd7 Rd2 Nb4 Qb3 Be6 Bc4 Nb6 Bxe6 Rxe6 Na4 Re4 Bg3 Nc4 Nxc5 Nxd2 Nxd2 Re2 Qxb4 a5 Qxb7 Rxd2 d7 Rxb2 Qd5 Rb5 Rd1 Bf8 Bd6 Bxd6 Qxd6 Rab8 h3 Rb1 Rxb1 Rxb1 Kh2 Rb6 Qe5 Kf8 Qh8 Ke7 Qe5 Kf8 f4 h5 Qd5 h4 Qe5 g5 Qh8 Ke7 Qe5 Kf8 fxg5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O Bg5 c5 d5 d6 Nd2 h6 Bh4 a6 e4 b5 Be2 b4 Na4 Nh7 O-O Nd7 Qc2 g5 Bg3 Ne5 Rae1 a5 Nf3 Ng6 e5 g4 Nh4 Nxe5 Nf5 Bxf5 Qxf5 Qc8 Qxc8 Raxc8 Bxe5 dxe5 Bxg4 f5 Bd1 Ng5 f3 Nf7 Bc2 Nd6 b3 e4 g3 Bd4 Kg2 e3 f4 Rfe8 Re2 Rb8 Kf3 e6 dxe6 Rxe6 Rg2 h5 Ke2 Kf7 h3 Rg6 Rfg1 Rg7 Bd3 Rbg8 Nb6 h4 g4 fxg4 Rxg4 Rxg4 hxg4 h3 g5 Rh8 Nd5 h2 Rh1 Ke6 Nc7 Kd7 Nd5 Ke6 Nc7 Kd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 cxd4 Nxd4 Bc5 Qd2 Bxd4 Bxd4 Nxd4 Qxd4 Qb6 Qd2 Nc5 O-O-O Bd7 Qd4 a6 h4 O-O-O Rh3 Bc6 Ne2 Kb8 Rc3 Na4 Qxb6 Nxb6 Nd4 Bd7 b3 Nc8 h5 Ne7 g4 Rdf8 g5 h6 Rh3 g6 hxg6 Nxg6 Rxh6 Nxf4 Bd3 Rhg8 Rg1 Rg7 Kd2 Rfg8 Bh7 Rh8 Ke3 Rhxh7 Rxh7 Rxh7 Kxf4 Rh4 Rg4 Rh1 Nf3 Rc1 Rg2 Kc7 a4 Kd8 Rh2 Be8 Ke3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 Nc3 Bg7 g3 e5 Bg2 d6 d3 Nc6 e4 Nge7 Nge2 O-O O-O Nd4 Nxd4 exd4 Nd5 Nxd5 cxd5 c5 dxc6 bxc6 Qc2 Bb7 Bd2 a5 a3 Ba6 b4 axb4 axb4 Bb5 Rfc1 Rxa1 Rxa1 Qd7 Qb3 h5 h3 Kh7 Bf4 Be5 Bxe5 dxe5 Ra5 h4 Qd1 Kg7 Qd2 Rh8 Kh2 Qe7 Kg1 g5 Ra7 Qf6 Qd1 Rh6 Bf1 Qe6 Qd2 Rg6 Qd1 g4 hxg4 hxg3 fxg3 Qf6 Ra2 Qg5 Rf2 Qxg4 Qxg4 Rxg4 Kg2 Rg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Bd6 O-O Nc6 Nxc6 bxc6 c4 O-O c5 Be7 f3 Ng5 Nc3 Re8 Qa4 Bd7 Bd2 Rb8 Rab1 Bf6 Kh1 h5 Ne2 h4 h3 Qc8 Ba6 Qd8 Bd3 Qc8 Ba6 Qd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nf3 Be7 Bc4 O-O O-O Be6 Qe2 b5 Bb3 h6 Rfd1 Nbd7 a3 Qe8 Ne1 Rc8 f3 Nc5 Ba2 Bxa2 Rxa2 Ne6 Nd3 a5 Nc1 Nd4 Bxd4 exd4 Nxb5 d5 e5 d3 Qxd3 Bc5 Kh1 Qxe5 Nb3 Be3 Re1 Rfe8 N3d4 Bxd4 Rxe5 Rxe5 Ra1 Bxb2 Rb1 Rce8 g3 Re3 Qd2 Re2 Qd1 R8e3 Kg1 Be5 f4 Bb2 c3 Ne4 Qxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 b6 a3 Bf8 Nf3 Ne7 h4 h6 h5 a5 Bb5 c6 Ba4 Nd7 Ne2 b5 Bb3 c5 c3 Nc6 O-O Qc7 Re1 c4 Bc2 Nb6 Bf4 Be7 Bg3 Rb8 Nh2 Qd8 Ng4 b4 axb4 axb4 cxb4 Nxb4 Bb1 Bd7 b3 Ra8 Rxa8 Qxa8 bxc4 Nxc4 Nc1 Ba4 Qe2 Qa7 Ne3 Qxd4 Nxc4 dxc4 Qf1 O-O
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 c5 d5 a6 a4 h6 Be3 e6 Qd2 exd5 cxd5 Re8 Nge2 Nbd7 Nd1 Ne5 Nec3 Bd7 Be2 Kh7 O-O Rb8 a5 b5 axb6 Rxb6 Nf2 a5 Ra2 Rb4 Rfa1 Qb6 Rxa5 Rxb2 Qc1 Rb8 Bd2 Qb3 R5a3 Qc2 Qxc2 Rxc2 R3a2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 d5 d4 c6 Bg5 h6 Bh4 dxc4 e4 Bb4 Bxc4 Bxc3 bxc3 Qa5 O-O Nxe4 Ne5 Nd7 Qg4 g5 Qxe4 Nxe5 dxe5 gxh4 Rab1 h3 Rfd1 Rg8 g3 Rg5 f4 Rg8 Bf1 Qxc3 Bxh3 f5 Qe2 Kf8 Qh5 Qc5 Kh1 Qe7 Rd6 Kg7 Bxf5 Qf7 Qxf7 Kxf7 Bxe6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 d6 Bc4 Qb6 Nb3 e6 Bf4 Ne5 Be2 Bd7 a4 Be7 a5 Qc7 a6 O-O O-O Rfb8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 cxd5 exd5 Qc2 Na6 a3 Nc7 Bg5 g6 e3 Bf5 Bd3 Bxd3 Qxd3 Be7 O-O O-O b4 Ne4 Bf4 Nxc3 Bxc7 Qxc7 Qxc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 Bb3 Bg4 h3 Bh5 Nbd2 h6 Nf1 O-O g4 Bg6 Ng3 Re8 g5 Nh5 Nh4 Nxg3 fxg3 Kh7 gxh6 gxh6 Rf1 Rf8 Qg4 Qc8 Nf5 Bxf5 Rxf5 Ne7 Qh5 Nxf5 exf5 Qd8 Qxh6 Kg8 f6 Qxf6 Qxf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 e3 Bxc3 bxc3 c5 Bd3 Nc6 e4 d6 O-O e5 d5 Na5 Nd2 h6 Re1 g5 Nf1 Ba6 Ne3 Qd7 Qc2 O-O-O a4 Kb8 Qa2 Nh5 Nf5 Nf4 Bf1 Rdg8 Rb1 Kc7 f3 f6 Rb2 Ng6 Bd3 Nf4 Bf1 Ng6 Bd3 Nf4 Bf1 Ng6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 Bxc3 bxc3 c5 Bd3 O-O Ne2 Nc6 O-O b6 e4 h6 f4 d5 cxd5 exd5 exd5 Nxd5 dxc5 Re8 Qc2 f5 Bxf5 Bxf5 Qxf5 Rxe2 c4 Ne3 Qe6 Kf8 cxb6 Qd4 Ba3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 g6 d4 cxd4 Nxd4 Bg7 Nb3 d6 Be2 Nf6 O-O O-O Re1 Be6 Bf1 Rc8 Bg5 Re8 Nd5 Bxd5 exd5 Ne5 c3 a6 h3 Nc4 Bxc4 Rxc4 Qd3 Rc8 Rad1 Qc7 Re2 Qc4 Qxc4 Rxc4 Nd2 Rc7 Ne4 Nxe4 Rxe4 h6 Be3 Kf8 g4 Kg8 a4 Kf8 a5 Rec8 Rb4 Ke8 Kf1 Kd7 Rb6 Ke8 Ke2 Kd7 h4 Ke8 f3 Kd7 f4 Ke8 h5 Rd7 Kd3 Kd8 Rh1 Ke8 c4 Kf8 b3 Rdc7 Rh2 Ke8 Ke4 Kf8 hxg6 fxg6 f5 g5 Bd4 Rd7 Kd3 Kf7 Bxg7 Kxg7 Kd4 Rcc7 b4 Rc8 Rh3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Nf6 d4 Qxd5 Nf3 Bg4 Be2 e6 O-O Nc6 Be3 cxd4 cxd4 Be7 Nc3 Qd8 Ne5 Bxe2 Qxe2 O-O Nxc6 bxc6 b3 Qd6 Rac1 Rfd8 Rfd1 Qb4 Na4 Qb5 Rc4 Rac8 Rdc1 Ba3 R1c2 Qf5 h3 Nd5 Kh2 Bd6 g3 Nb4 Rd2 Rb8 Kg2 Rdc8 Nc5 Rb5 a3 Nd5 b4 Nb6 Rc1 Qd5 Qf3 Nc4 Rd3 a5 bxa5 Bxc5 Qxd5 exd5 dxc5 Rxa5 Rcc3 Rca8 Bc1 Rxc5
Result: 0-1
Processing game...
FEN: 
Moves: c4 g6 g3 Bg7 Bg2 d6 Nc3 c5 e3 Nf6 Nge2 O-O O-O Nc6 d3 Bd7 h3 a6 b3 Rb8 Bb2 b5 Qd2 Qb6 Rad1 bxc4 dxc4 Ne5 Ba1 Bxh3 f4 Bxg2 fxe5 Ne4 Nxe4 Bxe4 exd6 Bxa1 dxe7 Rfe8 Rxa1 Rxe7 Rad1 Qe6 Qd6 Qh3 Qxb8 Kg7 Kf2 Qg2 Ke1 Bf3 Rxf3 Qxf3 Qf4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 e6 Nc3 Bb4 cxd5 Nxd5 Bd2 O-O e4 Nxc3 bxc3 Be7 Bd3 c5 O-O Nc6 Be3 cxd4 cxd4 Qa5 a3 b6 d5 exd5 exd5 Nd8 Re1 Bf6 Rc1 Qxa3 Ng5 Bxg5 Bxg5 Ba6 Bxh7 Kxh7 Be7 Qd3 Qh5 Kg8 Bxf8 Nb7 Be7 Bc4 d6 Nd8 Bxd8 Rxd8 Qh4 Qxd6 Qxc4 a5 h3 Ra8 Qc6 Qxc6 Rxc6 b5 Rb6 b4 Ra1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 g3 Qc7 Bg2 e6 O-O Nc6 Be3 Be7 Nxc6 bxc6 Na4 Bb7 c4 c5 Nc3 e5 f4 exf4 gxf4 Qd7 Kh1 Qg4 Qxg4 Nxg4 Bg1 Rd8 Rad1 Nf6 b3 Bc6 Rfe1 Nh5 Be3 Bh4 Re2 g6 Red2 Be7 e5 Bxg2 Kxg2 dxe5 Rxd8 Bxd8 fxe5 Bc7 Rd5 O-O Kf3 Re8 Bxc5 Bxe5 Ne4 f5 Nd6 Re6 b4 Nf6 Rd2 f4 b5 axb5 cxb5 g5 b6 g4 Kg2 f3 Kf1 g3 hxg3 Bxg3 Rd1 f2 Bxf2 Bxd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 a6 Ne5 Nbd7 Bf4 e6 e3 Nxe5 Bxe5 Be7 c5 O-O Bd3 Nd7 Bg3 e5 b4 e4 Be2 f5 h4 g6 a4 Qe8 Bf4 Qf7 b5 axb5 axb5 Rxa1 Qxa1 Qg7 Qa5 Bd8 b6 h6 Kd2 g5 hxg5 hxg5 Bd6 Be7 Bxe7 Qxe7 Qa1 Nf6 Qg1 Rf7 g3 Rh7 Rxh7 Qxh7 Qa1 f4 Qa8 Qf5 gxf4 gxf4 Qb8 f3 Ba6 Qd7 Bf1 Kf7 Na2 Qd8 Bh3 Nd7 Qf4 Nf6 Qb8 Nd7 Bxd7 Qxd7 Nc1 Qh3 Qc7 Kf6 Qd8 Kf7 Qc7 Kf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Bd3 Nc6 Bc2 Bg4 h3 Bh5 d3 e6 Nbd2 Be7 Qe2 Qc7 Nf1 O-O g4 Bg6 Ng3 d5 h4 dxe4 dxe4 h6 h5 Bh7 g5 Ng4 Ng1 Bd6 Qxg4 f5 gxf6 Rxf6 N3e2 Raf8 f4 Bxf4 Nxf4 e5 Nf3 exf4 Bd2 Re8 O-O-O Bxe4 Rhe1 Ne5 Nxe5 Qxe5 Rxe4 Qxe4 Bxe4 Rxe4 Be1 Re8 Bf2 b6 Rd7 Rf7 Rxf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 d4 Nf6 Nc3 cxd4 Qxd4 e5 Qd3 h6 Nd2 Be7 Nc4 O-O Bxd7 Bxd7 Nxd6 Qc7 Nf5 Bxf5 exf5 e4 Nxe4 Qe5 f3 Rad8 Qc3 Qxf5 O-O Nxe4 fxe4 Qxe4 Be3 a6 Rf4 Qg6 Rf2 Bh4 g3 Bf6 Qb4 Rfe8 Bb6 Rd5 a4 h5 Qb3 Rde5 Raf1 h4 Kh1 Re1 a5 h3 Qf3 Rxf1 Rxf1 Qxc2 Qxb7 Re4 Rg1 Re2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 g3 d5 Bg2 g6 O-O Bg7 d4 O-O c4 dxc4 Na3 c3 bxc3 c5 e3 Nc6 Qe2 Qa5 Bb2 Nd5 Rfc1 Bg4 h3 Bxf3 Bxf3 Rfd8 Nc4 Qa4 Nd2 Rac8 Ne4 b6 c4 Nxe3 Qxe3 Nxd4 Kg2 Nc2 Rxc2 Qxc2 Bxg7 Kxg7 Ng5 h6 Qe5 Kg8 Ne6 fxe6 Qxe6 Kg7 Qe5 Kg8 Bd5 Rxd5 Qe6 Kg7 Qxe7 Kg8 Qe6 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Ba6 e3 d6 f3 Nbd7 Bd3 h6 Bh4 c5 Rd1 Rc8 Ne2 cxd4 Qxd4 Ne5 b3 Nxd3 Qxd3 d5 Qc3 Ne4 Qxg7 Kxg7 Bxd8 Rfxd8 fxe4 dxc4 bxc4 Rxd1 Kxd1 Rxc4 Kd2 Ra4 Ra1 Bxe2 Kxe2 b5 Rb1 a6 Rb3 Rxe4 Rc3 Ra4 h3 h5 g3 f5 Rb3 Kf6 Rc3 Ke5 Kf3 Kd5 Rd3 Kc4 Rd6 Kb3 Rxe6 Kxa3 Ke2 a5 Re5 b4 Rb5 b3 Kd3 Kb2 h4 Ra1 Rxf5 a4 Rxh5 a3 Ra5 a2 h5 Rh1 Ke4 a1=Q
Result: 0-1
Processing game...
FEN: 
Moves: c4 c6 d4 d5 Nf3 Nf6 Nc3 a6 c5 g6 Bf4 Bg7 h3 O-O e3 Nbd7 Bd3 Ne8 Rc1 f6 e4 e5 dxe5 Nxc5 exd5 fxe5 Be3 Nxd3 Qxd3 e4 Qxe4 Nf6 Qc4 Nxd5 Nxd5 Be6 O-O Bxd5 Qg4 Bxf3 gxf3 Qd5 Rcd1 Qxa2 Rd7 Rf7 Rfd1 Qb3 R1d3 Qxb2 Qc4 Rf8 Rxf7 Rxf7 Rd8 Bf8 Bh6 Qa3 Qe6 Qc5 h4 Qb4 f4 Qb1 Kh2 Qb4 Kg2 Qa3 h5 gxh5 f5 Qb4 Rxf8 Qxf8 Bxf8 Kxf8 f6 Rxf6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nf6 Bg2 Nc6 O-O Be7 Nc3 O-O Bg5 cxd4 Nxd4 Qb6 Nb3 Be6 Bxf6 Bxf6 Nxd5 Bxd5 Qxd5 Rfd8 Qf5 Bxb2 Rab1 Ba3 e3 Rac8 h4 Qc7 Rfd1 h6 h5 Ne7 Qe4 b5 Nd4 a6 Qb7 Rd6 Rb3 Qxb7 Bxb7 Rc7 Be4 Bc5 Rbd3 Bb6 Nf5 Rxd3 Rxd3 Rc1 Kg2 Nxf5 Bxf5 a5 Rd7 b4 Be4 Kf8 Bd5 Rc7 Rxc7 Bxc7 Bb3 Be5 f4 Bf6 Kf3 Bd8 e4 g6 hxg6 fxg6 g4 h5 gxh5 gxh5 e5 Bc7 Ba4 Kg7 Bc6 Kg6 Ke4 Bb6 f5 Kg5 f6 Bc5 Kd5 Bf8 Be8 h4 Bd7 Kg6 Ke6 h3 Be8 Kg5 Kf7 Bc5 e6 Bd4 e7 Bxf6 Bd7 Bxe7 Bxh3 a4 Be6 Kf4 Ke8 Ke4 Bg8 Ke5 Kd7 Ke4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 g3 g6 Bg2 Bg7 O-O Nc6 Nc3 d6 d3 Nf6 Bg5 O-O Qd2 Rb8 Bh6 b5 Bxg7 Kxg7 a3 a5 Ng5 Bd7 Nce4 Nxe4 Nxe4 h6 e3 b4 axb4 axb4 Qe2 Qc8 Rfc1 Bg4 Qf1 Qc7 Ra6 Rb6 Rca1 Rxa6 Rxa6 Bc8 Ra1 Bb7 Qd1 Rc8 b3 Qb6 Ra2 d5 Qa1 Kg8 Nd2 e6 Nf3 Qc7 h3 Qd6 Nh2 e5 Ng4 h5 Nh2 Qd8 Nf3 Ra8 Rxa8 Qxa8 Qxa8 Bxa8 g4 hxg4 hxg4 f6 g5 Kf7 Nh2 Nb8 Ng4 Nd7 gxf6 Ke6 f4 exf4 exf4 Nxf6 Nxf6 Kxf6 Kf2 Kf5 Kg3 g5 Bh3 Kf6 fxg5 Kxg5 Bg2 Kf5 Kf2 Ke5 Ke2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 Qe2 h6 Ne4 Nxe4 Qxe4 Nf6 Qe2 Qc7 Bd2 b6 O-O-O Bb7 Kb1 Rd8 Rhe1 O-O Ne5 c5 f4 cxd4 g4 Nd5 g5 hxg5 Qh5 f5 Ng6 Nf6 Qxg5 Nh7 Qh5 Rf6 Rg1 Be4 Rg3 Rc8 Rc1 Qc6 a3 Rc7 Bb4 Bxb4 axb4 a5 Rcg1 Rxg6 Qxg6 Bxd3 Qxg7 Rxg7 Rxg7 Kf8 Rg8 Kf7 R1g7 Kf6 Rg6 Kf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 O-O c3 h6 Bxf6 exf6 g3 f5 Bg2 d6 O-O Nd7 Qc2 Re8 e3 Nf6 Ne1 h5 h4 c6 Nd3 Ne4 Nf4 Qe7 b4 Bd7 a4 Rac8 Rab1 b6 Qd3 Bh6 Rfe1 Nf6 Nf3 Ne4 c4 Qd8 Nd2 Nf6 b5 c5 dxc5 dxc5 Nd5 Be6 Rbd1 Bf8 Qb3 Bxd5 cxd5 Bd6 Nc4 Ng4 a5 Ne5 axb6 axb6 Ra1 Nxc4 Qxc4 Be5 Ra6 Qd6 Re2 Re7 Ra4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 g6 e4 Bg7 d4 d6 c4 Bg4 Be2 Bxf3 Bxf3 Nc6 d5 Nd4 Nc3 c5 Be3 Qb6 Rb1 Nxf3 gxf3 Bxc3 bxc3 Qa6 Qe2 Nf6 e5 dxe5 Bxc5 Nd7 Be3 b6 c5 Qa5 O-O Nxc5 Rb5 Qa4 Bxc5 bxc5 Qxe5 O-O Rb7 Qxa2 Rxe7 Rad8 d6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 e6 d4 Nf6 c4 d5 Nc3 Be7 Bf4 O-O e3 a6 c5 Nh5 Bd3 Nxf4 exf4 Nc6 a3 f5 h4 Bf6 Ne2 Bd7 Rc1 Qe7 h5 Nd8 Ne5 Bb5 Qb3 Bxe5 fxe5 Qg5 g3 Nc6 f4 Nxd4 Qb4 Nxe2 fxg5 Bxd3 Rd1 Bb5 a4 Nxg3 Rh3 Ne4 axb5 f4 bxa6 bxa6 Qb7 Rab8 Qc6 Nxg5 h6 Rxb2 Rh5 gxh6 Rxg5 hxg5 Qxe6 Kg7 Qe7 Kg8 Qxg5 Kh8 e6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 e4 c6 h3 Re8 Re1 exd4 Nxd4 Qb6 Nb3 Ne5 Be3 c5 Bf1 Be6 Nd5 Bxd5 exd5 Ne4 Kg2 f5 Qc2 Nf7 Bd3 Re7 Re2 Rae8 Rae1 Qa6 Nc1 Nf6 Bd2 Ne5 Bc3 b6 Re3 Qb7 R1e2 b5 b3 b4 Bb2 Nh5 Bxe5 Bxe5 Rf3 Nf6 Rfe3 f4 Rf3 Nh5 Re4 fxg3 fxg3 Rf7 Ree3 Ref8 Ne2 Bd4 Bxg6 hxg6 Qxg6 Ng7 Rxf7 Qxf7 Qxf7 Rxf7 Nxd4 cxd4 Rd3 Nf5 Kf3 Ne3 Ke4 Re7 Kf4 Ng2 Kf5 Ne3 Kf4 Nf1 g4 Kg7 Rxd4 Re2 Re4 Rxa2 Re7 Kf8 Rd7 Rf2 Kg5 Nd2 Rxa7 Ne4 Kg6 Rf6 Kh5 Rf3 h4 Rxb3 g5 Rh3 Kg4 Rh1 Rb7 Rg1 Kf5 Ng3 Ke6 Rb1 Kxd6 b3 Kc6 Nf5 d6 b2 Rb8 Kf7 d7 Nd4 Kc5 Ne6 Kc6 Rc1 Rxb2 Rxc4 Kd6 Rd4 Ke5 Rxd7 g6
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Qxd4 Bd7 c4 Nc6 Qd2 g6 Be2 Bg7 O-O Nf6 Nc3 O-O Rb1 a6 b3 Qa5 Bb2 Rfc8 Rfd1 Bg4 Qe3 Nd7 Nd5 Bxb2 Rxb2 Bxf3 Bxf3 e6 Nc3 Rd8 Rbd2 Nde5 Be2 Nb4 h4 b5 cxb5 axb5 Nxb5 Nbc6 a3 d5 exd5 Rxd5 Rxd5 exd5 b4 Qa4 Rxd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bc4 Nf6 d3 e6 Bb3 Nc6 c3 d5 e5 Nd7 d4 Be7 O-O b5 Re1 Bb7 Nbd2 a5 a4 b4 c4 dxc4 Nxc4 cxd4 Bg5 O-O Bxe7 Qxe7 Nxd4 Nc5 Nxc6 Bxc6 Rc1 Bd5 Ba2 Qg5 Ne3 Bxa2 Rxc5 Rfc8 Rb5 Qd8 Qf3 Qd2 Rd1 Qxb2 h3 Rf8 Rd7 Rac8 Ng4 Bd5 Qg3 Qc1 Kh2 Bc6 Ra7 Bxb5 axb5 Kh8 b6 Qc5 Qf4 Qxb6 Rxf7 Rxf7 Qxf7 b3 Nf6 Qc7 Qxe6 gxf6 Qxf6 Qg7 Qe6 Rg8 g3 b2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 d6 g3 g6 Bg2 Bg7 O-O O-O Nc3 c6 d4 Bf5 Ne1 e5 d5 cxd5 cxd5 Na6 e4 Bd7 Nd3 Qe7 a4 Rfc8 Bd2 Nc5 Nxc5 Rxc5 a5 Qd8 Qb3 Ng4 h3 Bh6 Bxh6 Nxh6 Qxb7 Rb8 Qxa7 Rxb2 Na4 Rxa5 Nxb2 Rxa7 Rxa7 Bb5 Rfa1 Qb6 Nd1 Kg7 Ne3 Ng8 Ra8 Nf6 Rb1 Qc5 Ra5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 a4 Nc6 O-O Be7 Be3 O-O f4 Qc7 Kh1 Re8 Qd2 Bd7 Rad1 Rad8 Nb3 Bc8 Bf3 b6 Qf2 Nd7 Nd4 Bb7 Bh5 Rf8 Qg3 Nxd4 Bxd4 Bf6 Be2 e5 fxe5 Bxe5 Qf2 Nc5 Bf3 Rfe8 h3 a5 Rfe1 Bc6 b3 h6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 a4 Nc6 Be2 e6 Be3 Be7 f4 O-O O-O Qc7 Nb3 b6 Bf3 Bb7 Qe2 Nd7 Rfd1 Rab8 g3 Rfd8 Bg2 Nc5 Nd4 Bf6 f5 Nxd4 Bxd4 Nd7 Qf2 Bc6 g4 Bxd4 Rxd4 Ne5 h3 b5 axb5 axb5 f6 Be8 fxg7 b4 Nd1 Qc5 b3 Ra8 Rxa8 Rxa8 Rd2 Ra1 Kh2 Qxf2 Nxf2 Rc1 Kg3 Kxg7 Kf4 h6 Nd1 Kf6 Ne3 Ke7 Bf1 Bd7 h4 Ra1 g5 Ng6 Kg4 hxg5 Kxg5 Ne5 Kf4 f6 h5 Nf7 Kg3 Ra8 Rd4 Rg8 Kf2 Rb8 Be2 Ng5 Bf3 Nxf3 Kxf3 Rh8 Rxb4 Rxh5 Rb7 Kd8 Rb8 Bc8 Nc4 Kc7 Ra8 d5 exd5 exd5 Na5 Rh2 c3 Rh3 Kf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 Qxd5 e3 c5 Bd2 Bxc3 Bxc3 cxd4 Bxd4 Nc6 Bxf6 gxf6 Ne2 Bd7 a3 Qe5 Nc3 f5 O-O-O O-O-O g3 Kb8 Be2 Ne7 Qd3 Qc7 Qd6 Bc6 Qxc7 Kxc7 Rhe1 Rxd1 Rxd1 Rd8 Rxd8 Kxd8 Kd2 Nc8 Kd3 Nd6 Kd4 b6 b4 Ke7 f4 h6 a4 f6 a5 Bd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Kh1 Nbd7 a4 b6 f3 Bb7 Be3 Qc7 Rf2 Rad8 Bf1 Nc5 Rd2 h6 Nxc5 dxc5 Bc4 Nh5 Nd5 Bxd5 Rxd5 Bg5 Bg1 Nf4 Rxd8 Rxd8 Qe1 a5 Rd1 Kf8 Bf2 Rd6 h4 Be7 Rxd6 Qxd6 g3 Ne6 c3 g5 h5 g4 fxg4 Bg5 Kg2 Ke7 Qe2 Qd2 Qxd2 Bxd2 Bxe6 Kxe6 Kf3 Kd6 Ke2 Bc1 Kd3 Kc6 Be1 Bg5 Kc4 Be3 b4 cxb4 cxb4 axb4 Bxb4 Bc1 Bf8 Bg5 Bg7 f6 Bh8 Kd6 Kb5 Kc7 Bg7 Kb7 Bf8 Kc7 Be7 Kd7 Bb4 Be3 g5 fxg5 g4 Ke6 Kc6 Bd4 Bd6
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nc3 g6 g3 Bg7 Bg2 Nc6 a3 a6 Rb1 Rb8 b4 cxb4 axb4 b5 cxb5 axb5 Nf3 d5 d4 e6 Bf4 Rb6 e4 Nf6 e5 Ne4 Nxe4 dxe4 Nd2 O-O Be3 f5 exf6 Bxf6 Nxe4 Bxd4 O-O e5 Bg5 Ne7 Rc1 Bb7 Qc2 Rf7 Qd2 Qa8 Qd3 Nd5 Qb3 Nc3 Rxc3 Bxc3 Nf6 Rxf6 Bxb7 Qxb7 Bxf6 Qf3 Qe6 Qxf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 Nc3 Nxc3 dxc3 Be7 Be3 O-O Qd2 Nc6 O-O-O Ne5 Nd4 c5 Nb5 Qa5 a3 Be6 Nxd6 Qb6 f4 Ng4 f5 Qxd6 Qxd6 Bxd6 Rxd6 Nxe3 fxe6 fxe6 Bd3 Rae8 Be4 Re7 Re1 Ng4 h3 Nf6 Bf3 Rfe8 Re5 b6 Bc6 Rc8 Bb5 Kf7 Bc4 Rce8 b4 Nd5 Rdxe6 Rxe6 Bxd5 Kf6 Rxe6 Rxe6 Bxe6 Kxe6 Kd2 c4 b5 Kd5 Ke3 Ke5 h4 h5 g3 g6 Kf3 Kf5 a4 Ke5 g4 Ke6 gxh5 gxh5 Kf4 Kf6 Ke4 Ke6 a5 bxa5 Kd4 Kf5 Kxc4 Kg4 Kb3 Kxh4 c4 Kg5 c5 h4 c6 h3 c7 h2 c8=Q h1=Q Qd8 Kf5 Qd3 Ke5 Qe3 Kd6 Qd4 Ke6 Qc4 Kd6 Qf4 Ke6 Qe3 Kd6 c4 Qb1 Kc3 Qa1 Kd2 Qb2 Ke1 Qb4 Kd1 Qxc4 Qh6 Kc5 Qe3 Kxb5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O Bb3 d5 exd5 Nxd5 h3 a5 a3 a4 Ba2 Kh8 Re1 f6 d4 exd4 Nxd4 Ndb4 axb4 Qxd4 c3 Qxd1 Rxd1 Bf5 Na3 Ne5 Nb5 c5 bxc5 Bxc5 Nd4 Rfd8 Be3 Bg6 Rd2 Bf7 Bxf7 Nxf7 Re2 Bxd4 Bxd4 Nd6 Bb6 Rdc8 Re7 h6 Bd4 b5 Rae1 Nf5 Rb7 Nxd4 cxd4 Rcb8 Ree7 b4 Rxg7 Rxb7 Rxb7 a3 bxa3 bxa3 Rb1 a2 Ra1 Kg7 Kf1 Kf7 Ke2 Ke6 Kd3 Kd5 Kc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bd3 Nb4 a5 Bd7 Nf3 Bc6 Bb6 Qc8 Qe1 Nd7 Bd4 Nc5 Qg3 f6 e5 Rf8 Bxc5 dxc5 Bc4 Bd5 Nxd5 exd5 Bb3 c4 Ba4 Nc6 c3 fxe5 Nxe5 Nxe5 fxe5 Qe6 Bc2 Rxf1 Rxf1 Rf8 Rxf8 Bxf8 Qf4 g6 Bd1 Qf7 Qd4 Qf1 Qg1 Qxg1 Kxg1 Kf7 Bg4 b6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 Nc6 cxd5 Qxd5 e3 e5 Nc3 Bb4 Bd2 Bxc3 bxc3 Qd6 Bd3 Nf6 f3 O-O Ne2 Be6 O-O Rad8 Qc1 Nd7 Be1 Qe7 Qb1 Nb6 Bxh7 Kh8 Bd3 Na5 e4 Nac4 f4 f6 f5 Bf7 Rf3 Kg8 Rh3 exd4 cxd4 c5 d5 Ne5 Bc2 Nbc4 Nf4 b5 Qb3 Ng4 Qg3 Nce5 Bd1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nf3 Be7 Bc4 O-O O-O Be6 Bb3 Nc6 Bg5 Na5 Bxf6 Bxf6 Nd5 Nxb3 axb3 Bg5 Qd3 Bh6 Rad1 Rc8 Nd2 b5 c3 Kh8 b4 Ra8 Nb3 f5 Na5 Rc8 Qh3 Rf7 Rfe1 Qe8 Qh4 Qf8 exf5 Bxf5 f3 Bc2 Ra1 Bf5 h3 Qe8 Qf2 Qe6 Rad1 Qg6 Kh1 Rcf8 h4 Bc2 Rd2 Bxd2 Qxd2 Qg3 Nb7 Rxb7
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 Bb4 g3 O-O Bg2 d5 Qb3 c5 O-O Nc6 d3 h6 e3 Re8 a3 dxc4 dxc4 Bxc3 Qxc3 e5 b4 e4 Nd2 Qe7 b5 Ne5 Nxe4 Nf3 Bxf3 Nxe4 Bxe4 Qxe4 f3 Qe7 e4 Be6 Be3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bd3 Nb4 a5 Bd7 Nf3 Rac8 Qe2 Bc6 Bb6 Qb8 Nd4 Nxd3 cxd3 d5 Qf3 Nd7 Nxc6 bxc6 Na4 Qd6 Qe3 Qb4 Rfc1 c5 Qf3 Nf6 Nc3 Bd8 exd5 exd5 Nxd5 Nxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Nbd2 Nc5 c3 d4 Ng5 dxc3 Nxe6 fxe6 bxc3 Qd3 Nf3 O-O-O Qe1 Nxb3 axb3 Kb7 Be3 Be7 Bg5 h6 Bxe7 Nxe7 Nd4 Rxd4 cxd4 Qxb3 Qe3 Qxe3 fxe3 Nd5 Kf2 Kb6 Ke2 a5 Rf7 a4 Kd2 c5 e4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bd3 Nb4 a5 Bd7 Nf3 Rac8 Bb6 Qb8 e5 dxe5 fxe5 Nfd5 Nxd5 exd5 Re1 h6 c3 Nxd3 Qxd3 Bc5 Qxd5 Be6 Qd2 Bxb6 axb6 Rc6 Ra4 Rxb6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g5 Ba3 d6 exd6 Qxe2 Bxe2 Bg7 cxd5 Bxe2 Kxe2 Bxa1 Rc1 O-O-O Rxc6 Rhe8 Kd3 Rd7 Nc3 Bxc3 Kxc3 Re5 Kc4 Re4 Kd3 Re5 Kc4 Re4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Bd7 Nb3 Na5 Nxa5 Qxa5 Qd3 Rad8 Rfd1 Bc6 b4 Qc7 b5 Bd7 Rab1 axb5 Nxb5 Bxb5 Qxb5 Ra8 c4 e5 Bb6 Qc8 fxe5 dxe5 a5 Bf8 h3 Qe6 Rd5 Nxd5 exd5 Qg6 c5 e4 Be2 Re5 Qd7 Rg5 Rg1 e3 d6 Rg3 Qxb7 Qe6 Kh2
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Nbd2 Nc5 c3 d4 Ng5 dxc3 Nxe6 fxe6 bxc3 Qd3 Bc2 Qxc3 Nb3 Nxb3 Bxb3 Nd4 Qg4 Qxa1 Bxe6 Rd8 Bh6 Qc3 Bxg7 Qd3 Bxh8 Qg6 Bf6 Be7 Bxe7 Qxg4 Bxg4 Kxe7 Rc1 c6 f4 a5 Kf2 a4 Ke3 b4 Bd1 a3 g4 Rd5 Rc4 c5 Ke4 Rd8 Rxc5 Ne6 Rd5 Rc8 f5 Rc4 Ke3 Nc5 g5 Rc1 Rd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Ne5 Bb3 Rc8 h4 h5 Kb1 Nc4 Bxc4 Rxc4 Nde2 b5 Bh6 Qa5 Bxg7 Kxg7 Nf4 Rfc8 Ncd5 Qxd2 Rxd2 Nxd5 Nxd5 Kf8 Re1 Rb8 b3 Rc5 Nf4 Rbc8 Kb2 a5 a3 Kg7 Nd5 Be6 b4 axb4 axb4 Rc4 Nb6 Rxb4 Ka3 Rxc2
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bc5 a4 Bb7 d3 d6 Nc3 b4 Nd5 Na5 Nxf6 Qxf6 Ba2 h6 c3 bxc3 bxc3 O-O Be3 Rad8 Rb1 Bc8 Qe2 Be6 h3 Bxa2 Qxa2 Bxe3 fxe3 Qe6 Qxe6 fxe6 Rb4 Rb8 Rfb1 Nc6 Rb7 Rbc8 Kf2 Rf7 Ke2 Rcf8 d4 g5 Kd3 Rg7 d5 exd5 exd5 g4 dxc6 e4 Kxe4 gxf3 gxf3 Re7 Kd4 Rxf3 e4 Rxh3 Rxc7 Rxc7 Rb8 Kf7 Rb7 Re7 c7 Rxc7 Rxc7 Ke6 Ra7 h5 Rxa6 Rh1 Ra8 h4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 Qd2 Nc6 f3 O-O Bc4 Bd7 h4 h5 Bb3 Rc8 Nxc6 bxc6 Bh6 c5 Bc4 Qb6 Bxg7 Kxg7 b3 Be6 Nd5 Bxd5 exd5 e5 dxe6 d5 Be2 c4 c3 Rce8 bxc4 Rxe6 Kf1 Rfe8 Bd3 dxc4 Bxc4 Ne4
Result: 0-1
Processing game...
FEN: 
Moves: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Ne5 Be6 Bd3 Nbd7 f4 g6 O-O Bg7 Kh1 Bf5 Bc4 e6 Be2 h5 Be3 Rd8 Bg1 O-O Bf3 Nd5 Nxd5 exd5 Bf2 Qc7 Rc1 f6 Nd3 Rfe8 b3 Nb6 a4 Nc8 c4 Qf7 a5 Bf8 cxd5 cxd5 Bh4 Nd6 a6 b6 Ne5 Qe6 g4 hxg4 Nxg4 Bg7 Rc7 Ne4 Ne3 Bh3 Rg1 g5 Bg4 Bxg4 Qxg4 Qxg4 Rxg4 Nd6 Bf2 Nb5 Rb7 Re4 f5 Rxg4 Nxg4 Rc8 Rd7 Rc2 Rxd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 g4 Be6 O-O-O Nxd4 Bxd4 Qa5 Kb1 Rfc8 a3 Rab8 Nd5 Qxd2 Rxd2 Nxd5 Bxg7 Ne3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bd3 Nb4 a5 Bd7 Nf3 Rac8 Bb6 Qb8 Bd4 Bc6 Qd2 Nxd3 cxd3 Nd7 Bg1 Qc7 Nd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 h4 h5 O-O-O Rc8 Bb3 Ne5 Bg5 Rc5 Kb1 Re8 Rhe1 Qa5 a3 b5 Bxf6 exf6 Nde2 Rc6 Nd5 Qxd2 Rxd2 Nc4 Bxc4 bxc4 Red1 f5 exf5 Bxf5 Nd4 Bxd4 Rxd4 Re2 R4d2 Rxd2 Rxd2 Kf8 Kc1 Be6 Rd4 Bxd5 Rxd5 Ke7 Rb5 Ke6 Rb7 Rc5 Rxa7 g5 Ra8 gxh4 Re8 Kd7 Re4 c3 Rxh4 cxb2 Kxb2 Rg5 a4 f5 a5 f4 a6 Kc7 Rxf4 Rxg2 Rf7 Kb8 Kc3 h4 Kd3 Rf2 c4 Ra2 Ke4 Rxa6 Rh7 Ra5 f4 Kc8 f5 Kd8 Kf4 Rc5 Kg5 Rxc4 Kg6 Rg4 Kf7 d5 f6 Kd7 Kf8 Ke6 f7 Rf4 Kg8 d4 f8=Q Rxf8 Kxf8 Ke5 Rxh4 d3 Rh3 Ke4 Rxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 Be2 O-O Nf3 e5 d5 a5 Bg5 h6 Bh4 Na6 Nd2 Bd7 O-O Qe8 Kh1 Nh7 a3 h5 f3 Bf6 Bxf6 Nxf6 b3 Qe7 Qe1 Kg7 Qf2 h4 f4 exf4 Qxf4 h3 gxh3 Bxh3 Rf3 Rh8 Re3 Rae8 Rg1 Qe5 Qxe5 Rxe5 Nf3 Reh5 Nd4 Rh4 Bf3 Nc5 Rg3 Bd7 Rg2 Rh3 Rge2 Ng4 Bxg4 Bxg4 Rxh3 Rxh3 Rc2 Rd3 Ndb5 Bh3 Kg1 c6 b4 axb4 axb4 cxb5 bxc5 b4 Ne2 Rf3 Ng3 dxc5 d6 Kf8 e5 Ke8 Re2 b3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 e6 Bd3 Be7 N1f3 h6 Ne4 Ngf6 c3 O-O O-O c5 Ne5 cxd4 cxd4 Qb6 Nxf6 Bxf6 Be3 Rd8 Qc2 Nf8 Rfd1 Bd7 d5 Qd6 Nxd7 Rxd7 dxe6 Qxe6 Bc4 Rc7 Bxe6 Rxc2 Bd5 Rxb2 Rab1 Rxb1 Rxb1 Rd8 Bxb7 Bd4 Bxd4 Rxd4 g3 Ne6 Bc6 g6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 c3 c5 e3 Qb6 Qb3 O-O Nbd2 d6 Qxb6 axb6 Nc4 Nbd7 Be2 d5 Nce5 h6 Bf4 c4 h4 b5 g4 Nb6 g5 hxg5 hxg5 Ne4 Nd2 Bf5 Bg4 Na4 Nxe4 dxe4 Bxf5 gxf5 Rb1 b4 cxb4 Nb6 a3 Rfc8 g6 Bxe5 gxf7 Kxf7 Bxe5 Nd5 Kd2 c3 Kc2 cxb2 Kxb2 Rc3 Ra1 b5 Rhc1 Rd3 Rc5 Nb6 Rxb5 Nc4 Kc1 Rg8 Bg3 f4 Rf5 Kg6 Rxf4 Rc8 Rg4 Kh5 Rh4 Kg6 Rg4 Kh5 Rh4 Kg5 f4 Kg6 f5 Kg5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3 d6 Be3 Nf6 f3 Be7 Qd2 O-O g4 a6 O-O-O Nxd4 Bxd4 b5 Kb1 Bb7 h4 Rc8 g5 Nd7 a3 Ne5 Qe3 Nc4 Bxc4 Rxc4 Rhg1 b4 axb4 Rxb4 Rg2 Ba8 b3 a5 h5 Qc7 Bf6 Rfb8 Bxe7 Qxe7 Rd4 e5 Rxb4 axb4 Na2 Qc7 Qd3 Kf8 g6 fxg6 hxg6 h6 f4 exf4 Rf2 Kg8 Rxf4 Qc5 Rf1 Qb5 Rc1 Qe5 Qc4 Kh8 Rh1 Bxe4 Qc7 Rf8 Rxh6 Kg8 Nc1 Bxc2
Result: 0-1
Processing game...
FEN: 
Moves: d4 e6 c4 d5 Nc3 Be7 Nf3 Nf6 Bf4 O-O e3 c5 dxc5 Bxc5 Qc2 Nc6 a3 Qa5 O-O-O Be7 h4 dxc4 Bxc4 b6 Ng5 Ba6 Nce4 g6 Nxf6 Bxf6 Ne4 Be7 Bxa6 Qxa6 Kb1 Qb7 h5 Rac8 hxg6 Nb4 gxh7 Kh8 Be5 f6 Nxf6 Bxf6 Bxf6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O Bb3 d5 exd5 Nxd5 Re1 Bg4 h3 Bh5 g4 Bg6 Nxe5 Nxe5 Rxe5 c6 Qf3 Bf6 Re2 Kh8 Bxd5 cxd5 Nc3 Bxc3 bxc3 Rc8 Ba3 Re8 Rxe8 Qxe8 Qxd5 Rxc3 Bb4 Rxc2 Re1 Qc8 Qxb7 h5 Qxc8 Rxc8 d4 hxg4 hxg4 f5 gxf5 Bxf5 Bc5 a5 Re7 Rc6 Re5 Bb1 a3 Kg8 f3 a4 Kf2 Rf6 Re8 Kf7 Re7 Kg6 Ke3 Ba2 Ra7 Bd5 f4 Kf5 Rxg7 Re6 Kd3 Kxf4 Rf7 Kg4 Rd7 Bb3 Bb4 Kf3 Rf7 Kg4 Rd7 Kf3 Rf7 Kg4 Rf8 Rh6 Ke4 Re6 Kd3 Rh6 Bc5 Rh3 Ke4 Rh5 Rf4 Kg3 Ke3 Rh6 Rf1 Kg4 Ke4 Re6 Kd3 Re8 Kc3 Rb8 Rf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 d4 d5 Nc3 Be7 Nf3 Nf6 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6 Qc2 Qa5 O-O-O Be7 h4 Rd8 g4 Bd7 Kb1 dxc4 Bxc4 Rac8 g5 Nh5 Bd6 g6 Be2 Bxd6 Rxd6 Ne7 Qb3 Bc6 Rxd8 Rxd8 Rd1 Qf5 Ka1 Rf8 e4 Qc5 Qb4 Qxb4 axb4 Nf4 Bf1 a6 Ne5 Rc8 Ka2 Kg7 Kb3 Be8 Ng4 Ng8 Rd4 Bc6 b5 axb5 Bxb5 h5 Bxc6 Rxc6 Ne5 Rc7 Kb4 Ne7 Nb5 Rc2 Rd7 Nc6 Nxc6 bxc6 Nd6 Nd3 Ka5 Ne5 Rd8 Nc4 Nxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5 Nc6 fxe6 fxe6 Nxc6 bxc6 Be2 Be7 O-O O-O Rb3 Qc5 Be3 Qe5 Bd4 Qa5 Bb6 Qe5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5 Bg5 h6 Bh4 c5 dxc5 g5 Bg3 Ne4 e3 Qa5 Ne2 Nc6 a3 Bf5 Qc1 Nxc5 axb4 Nd3 Kd2 Qxb4 Ra4 Nxc1 Rxb4 Nxe2 Rxb7 Nxg3 hxg3 d4 exd4 Nxd4 Nd5 Rd8 Bc4 O-O Kc3 Ne2 Kb4 Nd4 Ne7 Kg7 Rxa7 Rb8 Kc5 Ne6 Bxe6 Bxe6 b4 Rfd8 Re1 Kf6 Rh1 Kg7 Re1 Kf6 Rh1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nc6 e4 e5 Bc4 Nf6 d3 Be7 O-O O-O Re1 d6 c3 h6 h3 Re8 Nbd2 Bf8 Nf1 g6 Ne3 Bg7 b4 Ne7 Bb3 c6 Bb2 a5 a3 Nh5 d4 Nf4 c4 Qc7 Qc2 Be6 dxe5 dxe5 c5 axb4 axb4 Rxa1 Bxa1 Rd8 Bxe6 Nxe6 Nc4 Nd4 Bxd4 exd4 Nd6 Rxd6 cxd6 Qxd6 Qc5 Qd8 Rd1 b6 Qc4 Qd6 Ne1 Qe5 Qc2 g5 Nd3 Qe6 Rc1 h5 Qe2 Ng6 Qxh5 Qxe4 Qf3 Qxf3 gxf3 Ne5 Nxe5 Bxe5 Rxc6 d3 Kf1 d2 Ke2 Bd4 Rd6 d1=Q Kxd1 Bxf2 Ke2 Bg1 Rd5 f6 Rd7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 f4 exf4 Nf3 d6 d4 g5 h4 g4 Ng5 h6 Nxf7 Kxf7 Bxf4 Bg7 Bc4 Ke8 O-O Nc6 Be3 Qxh4 Rf7 Rh7 e5 Na5 Bd3 Kxf7 Qf1 Ke7 Bxh7 Be6 Nd2 Rf8 exd6 cxd6 Qe2 Kd8 c3 Ne7 Re1 Bc4 Bf2 Rxf2 Qxf2 g3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 dxc4 Nc3 c6 a4 Bf5 Ne5 Nbd7 Nxc4 Qc7 g3 e5 dxe5 Nxe5 Bf4 Rd8 Qc1 Bd6 Nxd6 Qxd6 Bg2 Qe7 O-O a5 Nd1 O-O Qc3 Ng6 Bg5 Qxe2 Bxf6 gxf6 Ne3 Be6 Qxf6 Qh5 Qc3 Ne5 Rfd1 Nf3 Bxf3 Qxf3 Rxd8 Rxd8 Qxa5 Rd7 h4 h6 Qe1 Bh3 Kh2 Be6 a5 Qf6 b4 Rd3 Kg1 Rxe3 fxe3 Bd5 Rd1 Qf3 Rxd5 Qxd5 Qf2 Kg7 Kh2 h5 Qf4 Qb5 Kh3 f5 e4 fxe4 Qxe4 Qf1 Kh2 Qf2 Qg2 Qf5 Qd2 Qb5 Kg2 Kf7 Kf3 Kg6 Qd6 Kf7 Qf4 Kg6 Qd6 Kf7 Kf4 Qe2 Kg5 Qg4 Kh6 Qg7 Kxh5 Qh7 Kg5 Qg7 Kf4 Qc3 Qc5 Qd2 Ke5 Qd3 Qf2 Kg8 Qf4 Qd5 Kf6 Qd8 Kg6 Qe8 Kg5 Qe7 Kg4 Qd7 Qf5 Qd4 Kh5 Qxb4 g4 c5 Kg6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 Nbd7 g4 h6 h4 b5 Rg1 Nb6 g5 Nfd7 Nd5 Bxd5 exd5 hxg5 hxg5 Rc8 Na5 Nxd5 Nb7 Nxe3 Nxd8 Nxc2 Kd1 Nxa1 Nb7 Nc2 g6 Nd4 gxf7 Kxf7 Bd3 g5 Qg2 Nf6 Qxg5 Rcg8 Bg6 Kf8 Nxd6 Rh5 Qg2 Bxd6 f4 e4 Kc1 Rh6 f5 Nxf5 Qf2 Rhxg6 Rxg6 Rxg6 Qxf5 Kf7 Qf2 Ke6 Qb6 Rg5 Qxa6 e3 Kd1 Re5 Qc8 Kf7 Ke2 Nd5 Qg4 b4 Qc4 Ke7 Qh4 Kd7 Qg4 Kc7 Qc4 Kd7 Qg4 Re6 Qg7 Kc6 Qg2 Kc5 Qg5 Rg6 Qd8 Rg2 Kd3 Rd2 Ke4 e2 Qa5 Kc6 Qa8 Kb6 Qd8 Bc7 Qh4 Nf4 Qe1 Ng2
Result: 0-1
Processing game...
FEN: 
Moves: Nf3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O f4 h6 h4 Nxd4 Qxd4 hxg5 hxg5 Ng4 Be2 e5 Qg1 exf4 Bxg4 Bxg4 Qh2 f5 Qh7 Kf7 Rh6 Bxg5 Rhxd6 f3 Kb1 Qxd6 Rxd6 fxg2 Qh2 Bf3 Qg3 Rh8 Qxg5 Rh1 Nd1 g1=Q Rd7 Kf8 Qxf5 Kg8 Qxf3 Rf8 Qb3 Kh7
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 O-O Qe8 Nd2 Nh7 a3 f5 f3 Bd7 b4 axb4 axb4 Nxb4 Qb3 c5 dxc6 Nxc6 c5 Kh8 cxd6 Nd4 Qxb7 Rb8 Qa6 Rc8 Bc4 Rc6 Qa2 Rxd6 Bd5 g5 Bf2 g4 exf5 gxf3 Nxf3 Bxf5 Ne4 Bxe4 Bxe4 Nf6 Bb7 Ng4 Rae1 Qh5 Bg3 Nf5 Nxe5 Nxg3 Rxf8 Bxf8 Nxg4 Rd1 Rxd1 Bc5 Qf2 Bxf2 Nxf2 Ne2 Kh1 Nc3 Rd8 Kg7 Bf3 Qb5 Rd3 Qb1 Bd1 Qb2 Kg1 Qc1 h3 h5 Kh2 h4 Bf3 Qc2 Rd7 Kf8 Ng4 Qc1 Rd4 Nb5 Rd5 Qf4 Kh1 Nd4 Ne5 Qc1 Kh2 Qc7 Rc5 Nxf3 gxf3 Qb8 Kg1 Qb4 Nd3 Qd4 Kg2 Qxd3 Rc8 Kf7 Kf2 Qd7 Rc2 Qxh3 Ke3 Qg3 Rc4 h3 Rf4 Qxf4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Nc3 d6 h3 Bd7 Re1 Nf6 Bxc6 Bxc6 e5 dxe5 Nxe5 Rc8 d3 O-O Bg5 Re8 Qd2 Nd7 Nxc6 Rxc6 Bh6 Nf8 Bxg7 Kxg7 Ne4 b6 a4 Qd4 a5 Ne6 axb6 axb6 b3 b5 Ng5 Nxg5 Qxg5 Re6 Qd2 Rxe1 Qxe1 Qb2 Qb1 Qxb1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 e6 O-O Nge7 c3 a6 Be2 d5 exd5 Nxd5 d4 Be7 c4 Nf6 dxc5 Bxc5 Qxd8 Kxd8 Nc3 Ke7 Rd1 Rd8 Rxd8 Nxd8 a3 Bd7 b4 Bd6 Be3 h6 Rd1 Rc8 h3 Be8 Nd4 Nd7 f4 Bc7 Nb3 f6 c5 Nc6 Bf3 Rd8 h4 Ndb8 Rxd8 Nxd8 Kf2 Nbc6 g3 Bf7 Nd2 Bg6 h5 Bd3 Be2 Bh7 Nc4 Bg8 Bf3 Kd7 Ke2 Ne7 Kd3 Ndc6 Bf2 e5 Nb6 Kd8 Nbd5 Bxd5 Nxd5 Nxd5 Bxd5 exf4 gxf4 Bxf4 Ke4 Bc1 a4 Nxb4 Bxb7 Kc7 Ba8 Kb8 Bd5 f5 Kxf5 Nxd5 Bd4 Nf6 Kg6 Kc7 Be5 Kc6 Kxg7 Nd7 Bd4 Nxc5 Kg6 Ne6 Bf6 Kc5 Be7 Kc4 Kf5 Nf4 Kg4 Kb3 a5 Ka4 Bd8 Ne6 Bb6 Bg5 Kf5 Nf4 Kg4 Nd5 Kf5 Kb5
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 O-O Qe8 Nd2 Nh7 a3 h5 f3 Bd7 b3 f5 Rb1 Nc5 Nb5 Bxb5 cxb5 Bh6 exf5 gxf5 b6 cxb6 Nc4 Qg6 Be7 Rf7 Bxd6 Rg7 Rf2 e4 Bxc5 bxc5 fxe4 fxe4 d6 Nf6 Qf1 Nd7 Rd1 b5 Rd5 bxc4 Bxh5 Qe6 bxc4 Be3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 Qc7 Qf3 g6 Be3 b5 Bd3 Nbd7 O-O Bb7 a3 Bg7 Kh1 O-O Bg1 Nb6 Qe2 Nfd7 a4 b4 Nd1 Nc5 a5 Nbd7 c3 Nxd3 Qxd3 Nc5 Qe3 Nxe4 cxb4 Rfc8 Nc3 Nxc3 bxc3 e5 fxe5 dxe5 Ne2 f5 Rae1 f4 Qh3 Qc4 Bb6 Rf8 Ng1 e4 Rd1 Rf7 Bd4 Bxd4 cxd4 f3 gxf3 exf3 Rf2 Bd5 Qg3 Qxb4 Rdf1 Re8 Nh3 Re1 Nf4 Rxf4 Qg5 Rxf1 Rxf1 Qxd4
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nf3 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 b4 Bb7 Bb2 d6 e3 Nbd7 Be2 Qe7 d3 c5 O-O Rac8 b5 Rfe8 a4 a5 bxa6 Bxa6 Rfb1 Ra8 Qc2 Bb7 Bc3 Bc6 a5 bxa5 Bxa5 Rec8 h3 h6 Ra2 Qe8 Rba1 Ra6 Nd2 Rca8 Bc7 Qc8 Bxd6 Bxg2 Kxg2 Rxa2 Rxa2 Rxa2 Qxa2 Qc6 Nf3 Qxd6 Qa8 Kh7 Qb7 g5 d4 Kg7 Kf1 e5 Qb2 exd4 exd4 Qe6 Kg2 Nh5 d5 Qf6 Qxf6 Kxf6 Bf1 Kf5 Nd2 Ne5 Nb3 Nd7 Kf3 Ke5 Ke3 f5 Na5 Nf4 h4 Kd6 hxg5 hxg5 f3 Kc7 Nc6 Kd6 Nd8 Ke5 Nc6 Kd6 Nd8 Ke5 Nc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 h6 Bxf6 Qxf6 Nf3 Qd8 g3 Qc7 Bh3 O-O-O f5 Kb8 Ne2 Be7 Kb1 Ne5 Ned4 Nc4 Qf2 Qb6 b3 e5 Qe2 Rc8 f6 Bxh3 fxe7 exd4 Nxd4 Rhe8 Qh5 Be6
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 Ne5 Nbd7 Nxc4 Nb6 Ne5 Nbd7 Qb3 Nxe5 dxe5 Ng4 Qxb7 Nxe5 f4 Ng6 e4 Bd7 f5 Ne5 Bf4 f6 Bxe5 fxe5 Rd1 Rb8 Qxa7 Rb4 Be2 g6 Rf1 Bg7 fxg6 Rd4 Bh5
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 c3 d6 Bb3 a6 h3 h6 Nbd2 Be6 Nf1 d5 Qe2 O-O Ng3 Re8 O-O Bf8 Rd1 Qe7 Bc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Be7 d4 Na5 Be2 exd4 Qxd4 Nf6 e5 Nc6 Qh4 Nd5 Qg3 g6 O-O Nb6 c4 d6 Rd1 Nd7 Bh6 Ncxe5 Nxe5 Nxe5 Nc3 f6 c5 Nf7 cxd6 cxd6 Qe3 Nxh6 Qxh6 Bf8 Qe3 Kf7 Nd5 Be6 Nf4 Qe7 Re1
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 c3 O-O Nbd2 d6 e4 c5 dxc5 dxc5 Be2 Nc6 O-O Qc7 Qc2 Rd8 Rfe1 h6 Bh4 Nh5 Nc4 Be6 Ne3 Nf4 Bf1 Ne5 Nxe5 Bxe5 Nc4 Bxc4 Bxc4 b5 Bf1 c4 Red1 Nh5 Bg3 Nxg3 hxg3 Qb6 a4 a6 axb5 axb5 Rxa8 Rxa8 g4 e6 Be2 Ra2 Kf1 Bb8 Qd2 Kg7 g5 h5 g3 Be5 Qd7 Ra8 Qe7 Ra7 Qe8 Bd6 Kg2 Be7 Rh1 Qb7 Bf3 Ra8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nc6 O-O e5 d5 Ne7 e4 Ne8 Ne1 f5 Nd3 Nf6 Bg5 h6 Bxf6 Rxf6 f4 exf4 Nxf4 Kh7 Qd3 Rf8 Rae1 Be5 exf5 Nxf5 Kh1 Qf6 c5 Qg7 cxd6 cxd6 Ne6 Bxe6 dxe6 Rac8 Nd5 Ne7 Ne3 Bxb2 Rxf8 Qxf8 Rf1 Rc1 Nd1 Bf6 Qxd6 Kg7 Bxb7 Nf5 Qxf8 Kxf8 g4 Nd6 Rxf6 Kg7 Rf1 Nxb7 Rf7 Kg8 Rxb7 Rxd1 Kg2 Re1 Rxa7 Rxe6 a4 Re2 Kg3 g5 h4 Re3 Kf2 Re4 Kf3 Rf4 Kg3 h5 gxh5 Rxh4 a5 Rxh5 a6 Rh6 Kg4 Rc6 Kxg5 Rb6 Kf5 Rc6 Ke5 Rb6 Kd5 Rf6 Kc4 Rf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 O-O e5 d3 Re8 Rb1 a5 a3 c6 b4 axb4 axb4 d5 cxd5 cxd5 Bg5 Be6 e4 d4 Nd5 Nbd7 Nxf6 Bxf6 Bxf6 Qxf6 Qd2 Kg7 Ne1 g5 f4 gxf4 gxf4 Qh6 Qf2 f6 Rc1 Kh8 Rc7 Ra2 Rc2 Ra1 Rc7 exf4 Rxd7 f3 Qxd4 fxg2 Nxg2 Rxf1 Kxf1 Qc1 Kf2 Bxd7 Qxd7 Rg8 Qh3 Qd2 Kf1 Qxb4 Qe3 Qb2
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 c6 Nc3 e6 Bg5 h6 Bxf6 Qxf6 e3 Nd7 Bd3 dxc4 Bxc4 g6 O-O Bg7 Qc2 O-O Rad1 Qe7 Rfe1 Rd8 a3 b6 Ne2 Bb7 Nf4 Nf8 h3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Nb3 Qa3 Bxf6 gxf6 Be2 h5 O-O Nd7 Kh1 h4 h3 Be7 Rad1 b6 Qe3 Bb7 f5 Rc8 fxe6 fxe6 Bg4 Qb2 Rd3 f5 Rb1 Qxb1 Nxb1 fxg4 hxg4 h3 Rc3 hxg2 Kxg2 Rg8 Rxc8 Bxc8 Kf1 Rxg4 N1d2 e5 Qc3 Bb7 Qc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Kh1 Nxd4 Qxd4 Bc5 Qd3 h5 Bg5 b5 f4 Bb7 e5 Nd5 Nxd5 Bxd5 a4 Qc6 Bf3 Bxf3 Rxf3 bxa4 f5 Rb8 Raf1 O-O Bf6 Qb5 Rg3 g6 Qd1 exf5 Rxf5 Rb6 Qxh5
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 d6 Nc3 Nf6 Be2 O-O Nf3 e5 d5 Na6 Nd2 Ne8 O-O f5 exf5 gxf5 f4 c5 dxc6 bxc6 Nb3 e4 Be3 Nac7 Qd2 Be6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 h3 O-O Bg5 Na6 Bd3 e5 d5 c6 Nge2 Nc5 Bc2 cxd5 cxd5 a5 O-O Bd7 a3 a4 Qd2 Qb6 Rab1 Rfc8 Ng3 Qd8 Bh6 Bxh6 Qxh6 Qf8 Qe3 Be8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Kh1 b5 a4 Bb7 Nd5 bxa4 Rxa4 Bc6 Ra3 Nxe4 Na5 Nf6 Nxc6 Nxc6 Bc4 Nd4 Rh3 g6 Qd2 Nf5 Nxf6 Bxf6 Bd5 h5 Bxa8 Qxa8 Ra3 Rc8 c3 Qb7 Qd3 Rc6 Rd1 d5 Ra5 d4 cxd4 exd4 b4 Nh4 f3 Qxb4 Rxa6 Rxa6 Qxa6 Bg7 Qd3 Nf5 Bf4 Qb2 g3 Kh7 Rd2 Qb7 Qe4 Qb5 Qd3 Qb7 Qe4 Qb5 Kg1 Qa5 Qd3 Ne7 Rd1 Nd5 Bd2 Qa7 Re1 Qd7 Kg2 Qa7 Kg1 Qa2 Re2 Qa1 Re1 Qa4 Rc1 Nc3 g4 Qa5 Re1 Qc5 h3 Bf6 f4 h4 Qf3 Qb5 g5 Bg7 Re7 Qb1 Qf1 Qf5 Qe1 Qxh3 Rxf7 Qg4 Kh2 h3 Qf1 Ne4 Be1 d3 Qxh3 Qxh3 Kxh3 d2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 e6 Nc3 dxc4 Qa4 c6 Qxc4 b5 Qd3 Bb7 a3 a6 e3 c5 dxc5 Bxc5 Qxd8 Kxd8 Bd2 Ke7 Bd3 Nbd7 Ke2 Bd6 Rhd1 Rac8 Rac1 Nb6 Be1 Nc4 Rc2 Bxf3 gxf3 Ne5 h3 Nxd3 Rxd3 Rhd8 Rcd2 Bc7 Rc2 Bb6 Rxd8 Kxd8 Rd2 Ke7 Rd1 g6 f4 Rc4 f3 Nd7 b3 Rc6 Ne4 Rc2 Rd2 Rxd2 Bxd2 Bc5 Nxc5 Nxc5 Bb4 Kd6 Kd2 a5 Bxc5 Kxc5 Kd3 f6 h4 Kd5 b4 axb4 axb4 h6 e4 Kd6 Ke3 e5 fxe5 fxe5 Kf2 Ke6 Kg2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nxc6 Qf6 Qd2 dxc6 Nc3 Ne7 Qf4 Ng6 Qxf6 gxf6 Bd2 Rg8 Na4 Bd6 O-O-O Be6 Nc3 O-O-O g3 Bg4 Be2 Ne5 Bf4 Bh3 Bh5 Bc5 Rxd8 Rxd8 Nd1 Bg2 Re1 Nf3 Re2 Ng1 Bg4 Kb8 Rd2 Rxd2 Kxd2 Bxe4 Nc3 Bg2 Be3 Be7 Bh5 Nf3 Ke2 Ng1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 c6 O-O Bg7 h3 O-O exd6 exd6 Re1 Nc7 Bg5 Bf6 Bh6 Re8 Rxe8 Nxe8 Bb3 d5 c4 dxc4 Bxc4 Nd6 Bb3 Nf5 Qd2 Nd7 Nc3 Nxh6 Qxh6 Nf8 Rd1 Be6 d5 cxd5 Nxd5 Rc8 Qe3 b6 Ne5 Rc5 Ng4 Bxg4 hxg4 Kg7 f4 h6 f5 g5 Qe2 Nh7 Nxf6 Qxf6 Rd7 Re5 Rxf7 Qxf7 Qxe5 Qf6 Qc7 Kh8 Kh1 a5 Be6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 e6 Nc3 d5 d4 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7 Rc1 c6 Bd3 Nxc3 Rxc3 dxc4 Bxc4 Nd7 O-O b6 Bd3 c5 Be4 Rb8 Qa4 Bb7 Bxb7 Rxb7 Qc2 Rc8 Rc1 Rbc7 b4 e5 dxc5 bxc5 Rc4 e4 Qxe4 Qxe4 Rxe4 cxb4 Re8 Rxe8 Rxc7 Ne5 Nxe5 Rxe5 Rxa7 h5 h3 Rb5 g4 hxg4 hxg4 g6 Kg2 b3 axb3 Rxb3 g5 Kg7 Kg3 Rc3 Re7 Ra3 Rd7 Kf8 Rc7 Kg7 Rc8 Kh7 Rc7 Kg7 Re7 Kf8 Re5 Kg7 f4 Rd3 Kf3 Rd2 Ke4 Rd1 Rd5 Re1 Rd3 Kf8 Ke5 Kg7 e4 Re2 Rd7 Re1 Rc7 Re2 Ra7 Re1 Ra4 Re2 Kd6 Rd2 Ke7 Rf2 e5 Rf1 Ra6 Rxf4 e6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 e3 d6 f3 Nbd7 Nh3 c5 dxc5 bxc5 Be2 Qb6 O-O d5 Rad1 Bc6 Nf2 h6 Bh4 Ba4 Rd2 Bb3 Ng4 Nxg4 fxg4 Rab8 g5 hxg5 Bxg5 Rb7 Be7 Re8 Bh4 Nf8 Bg3 Rd8 Bh4 Rdd7 cxd5 Rxd5 e4 Rxd2 Qxd2 Ba4 Bh5 Be8 Bf2 Qb5 Qd8 Bc6 Bg3 Rd7 Qe8 Qxf1 Kxf1 Rd1 Bxd1 Bxe8 Bf2 Bb5 Be2 Bxe2 Kxe2 Nd7 Kd3 a6 Bg1 f5 exf5 exf5 Kc4 Ne5 Kxc5 Nd3 Kb6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 a4 Be7 O-O O-O Be3 Nc6 f4 Re8 Bf3 Rb8 Kh1 Qc7 g4 Nxd4 Bxd4 e5 fxe5 dxe5 Ba7 Ra8 g5 Rd8 Qe2 Ne8 Be3 Be6 Qf2 Rdc8 Rad1 Bc5 Bxc5 Qxc5 Nd5 Bxd5 Rxd5 Qxc2 Rxe5 Qxf2 Rxf2 g6 Rd5 Nc7 Rd7 Ne6 Bg4 Nxg5 h4 Nxe4 Rfxf7 Rc1 Kg2 Rc2 Kg1 Rc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 f4 Nc5 O-O Ncxe4 Nxe4 Nxe4 f5 e5 Qh5 Qe7 Qf3 Nc5 Nc6 Qc7 Bd5 a5 Bg5 Ra6 Nd8 f6 Nf7 Rg8 Be3 g6 Ng5 Rg7 fxg6 Rxg6 Bf7 Qxf7 Nxf7 Kxf7 Bxc5 dxc5 Rad1 Be7 Rd5 Bg4 Qe4 Kg7 Rfd1 Bxd1 Rxd1 Re6 Qf5 Kf7 Re1 b6 h4 Rg7 Kf1 Bd6 Kf2 Bc7 Kf3 Ke7 Re4 Kf7 Rg4 Re7 Ke4 Rxg4 Qxg4 Bd8 a4 Kf8 c3 Rg7 Qc8 Ke8 Qe6 Kf8 g4 Rf7 h5 Rg7 h6 Rg6 Qd5 Be7 Kf5 Rxh6 Qb7 e4 Qb8 Kf7 Qxb6 e3 Qe6 Ke8 Qxe3 Rg6 Qe4 Rg5 Kf4 Kd7 Qb7 Ke6 Qc8 Kf7 Qc7 h5 gxh5 Rxh5 Qxa5 Bd6 Ke4 f5 Kd5 Be7 Qc7 Rh6 a5 Rd6 Ke5 Rf6 Qc8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 d5 cxd5 cxd5 a6 g4 h5 h3 b5 Bg5 Qa5 Nd1 Qxd2 Kxd2 Nc5 Nf2 Bd7 gxh5 Nxh5 Be7 Rfc8 Bxd6 Nb7 Be7 f6 Nd3 Kf7 Ba3 a5 b4 Nd6 Kd1 axb4 Bxb4 Nc4 Kc2 Bf8 Nc3 Bxb4 Nxb4 Ra3 Nc6 Bxc6 dxc6 Rxc6 Nxb5 Rxf3 Bxc4 Rxc4 Kd2 Kg7 Rhc1 Rxe4 Rc7 Kg8 Rc8 Kh7 Rc7 Ng7 Nd6 Rf2 Kc3 Ree2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 Nc6 Bg5 e6 Nf3 Be7 c5 h6 Bf4 Ne4 Bb5 Nxc3 bxc3 Bd7 O-O O-O Rc1 Re8 Re1 Bf6 Rb1 b6 Ba6 Bc8 Bb5 Bd7 Ba6 Bc8 Bd3 bxc5 Ne5 Bd7 Rb7 Bxe5 dxe5 Rb8 Rxb8 Qxb8 Qg4 Kf8 Re3 Qd8 h4 Qa5 Rg3 Ke7 Qxg7 Kd8 Qxf7 Qxc3 Bb5 Qa5 Rg7 Ne7 Bxd7 Kxd7 Qf6 d4 Bxh6 c4 Bg5 Qc5 Rxe7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 O-O Be7 Bb3 O-O f4 Nc6 Be3 Nxd4 Bxd4 b5 e5 dxe5 fxe5 Ne8 Ne4 Bb7 Qd3 Qc7 c3 Rd8 Bc2 Bxe4 Qxe4 g6 b4 Ng7 g4 h5 h3 a5 a3 Rd7 Rf3 Qd8 Rb1 Bg5 Rbf1 axb4 axb4 Be7 Rb1 h4 Qe3 Ne8 Be4 Nc7 Bc6 Nd5 Qe4 Nxc3 Bxc3 Qb6 Kg2 Rc7 Ba8 Rc4 Qb7 Qxb7 Bxb7 Rb8 Ba6 Rb6 Ra1 Rcc6 Bxb5 Rxb5 Ra8 Kg7 Ra7 Kf8 Ra8 Kg7 Ra7 Kf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Nbd7 Qc2 dxc4 e4 c5 dxc5 Bxc5 Bxc4 a6 a4 Qc7 O-O Ng4 h3 Nxf2 Rxf2 Bxf2 Qxf2 Qxc4 Qg3 f6 Qxg7 Qc5 Kh1 Qf8 Qg4 Qf7 e5 Rg8 Qc4 f5 Bg5 h6 Bh4 Qg6 Rg1 Nf8 Qb4 Rg7 Ne2 b5 axb5 Rd7 Nf4 Qf7 Rc1 Bb7 bxa6 Bxf3 gxf3 Rda7 Rg1 Rxa6 Nh5 Qc7 Rg7 Ra1 Kg2 Qc2 Bf2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 d5 cxd5 cxd5 a6 g4 h5 h3 Nh7 Rg1 h4 Nc1 Kh8 Nb3 Bf6 g5 Be7 O-O-O f6 gxf6 Rxf6 Be2 Qf8 Kb1 g5 Bxg5 Nxg5 Qxg5 Rf7 Qd2 Rh7 Nc1 Nc5 b4 Nd7 Rdf1 Qh6 Qxh6 Rxh6 Nd3 Nb6 Nf2 Bd7 Ng4 Rh7 f4 Rc8 Kb2 exf4 Rxf4 Bg5 Rf3 Bd2 Kc2 Bxg4 Rxg4 Be1 Rg1 Bg3 Kb3 Rhc7 Nd1 a5 bxa5 Nd7 Rf5 Kg7 Rh5 Kf6 Rf1 Kg6 Rg1 Kf6 Rf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 d6 d4 cxd4 Nxd4 Nf6 Bg5 e6 Qd2 Be7 O-O-O Nxd4 Qxd4 a6 f4 b5 Bxf6 gxf6 Bd3 Qc7 Qe3 Qc5 Qg3 b4 Ne2 a5 Kb1 Kf8 Qh3 h5 Rc1 d5 exd5 Qxd5 f5 a4 Rhe1 Rb8 Nf4 Qd6 Qf3 e5 Nd5 Bd8 Be4 Qc5 c3 bxc3 Rxc3 Qd6 Rd1 Bb6 Nxb6 Qxb6 Rc2 Kg7 a3 Kh6 Rdd2 Bb7 Bxb7 Qxb7 Rc6 Rhc8 Rxf6 Kg5 Rxf7 Qxf3 gxf3 Rc3 Rg2 Kf4 f6 Rxf3 Re7 Rf8 f7 Kf5 Rg8 Kf6 Rxe5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 c5 d5 a6 a4 h6 Be3 e6 Qd2 exd5 cxd5 Re8 Nge2 Nbd7 Nd1 Ne5 Nec3 Nh5 Bxh6 Qh4 Nf2 Bxh6 Qxh6 f5 Qd2 Rb8 Be2 f4 g3 fxg3 hxg3 Qxg3 O-O-O Qxf2 Qg5 Bg4 Rdf1 Qd4 Rd1 Qf2 Rdf1 Qd4 Rd1 Qf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Kh1 Nc6 f4 a5 a4 Nb4 Bf3 Qb6 g4 exf4 Bxf4 Nd7 Nd4 g6 Bh6 Re8 Nf5 gxf5 gxf5 Bf6 Rg1 Kh8 Qd2 Ne5 Qg2 Ng6 fxg6 fxg6 Rgf1 Bg7 Bxg7 Kxg7 Rad1 Bd7 Qg3 Re5 Bg2 Nxc2 Nd5 Qd8 Qf2 Nb4 Nb6 Bg4 Qf7 Kh8 Nxa8 Bxd1 Rxd1 Qxa8 Qf6 Kg8 Rxd6 Re8 Rd7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O h3 Bb7 d3 d6 a3 Kh8 Nc3 Qd7 Bd2 Rab8 Ne2 Nd8 Ng3 Ne6 c3 c5 Bc2 Bd8 Be3 Bc7 d4 cxd4 cxd4 exd4 Bxd4 Nxd4 Qxd4 Rbe8 Nf5 Re6 Rad1 Qc6 Qd2 Qb6 Qg5 Ne8 Bb3 Rf6 Ne7 Qa7 Nd5 Bxd5 Rxd5 Bb6 Qd2 Nc7 Rd3 Bc5 e5 dxe5 Rxe5 Qb6 Rd7 Ne6 Bxe6 Rxe6 b4 Rd6 Rxd6 Bxd6 Rd5 Bb8 Ne5 Bxe5 Rxe5 h6 Qd7 Qf6 Qe7 Rd8 Qxf6 gxf6 Re7 Kg7 Ra7 Rd6 g4 f5 gxf5 Rd3 Rxa6 Rxh3 Ra5 Kf6 Kg2 Rd3 Rxb5 Rxa3 f3 Rb3 Kg3 Kg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 g4 h6 f4 e5 Nf5 Nc6 Qf3 g6 O-O-O gxf5 exf5 e4 Nxe4 Nxe4 Qxe4 Qe7 Qd3 Bg7 Bd2 O-O g5 hxg5 fxg5 Qe5 f6 Bf5 Qg3 Rfc8 Bc3 Nb4 Bxe5 Rxc2 Kb1 Re2 Ka1 Nc2 Kb1 Na3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 Ne5 e6 f3 c5 e4 cxd4 exf5 Bb4 Bxc4 Qd6 Bb5 Nc6 Nc4 Qc5 Bd2 O-O Na2 Bxd2 Qxd2 Ne7 Qb4 Qxb4 Nxb4 a6 Nb6 axb5 Nxa8 Rxa8 fxe6 bxa4 exf7 Kxf7 Kd2 Ke6 Rhc1 Kd6 b3 b5 bxa4 bxa4 Rc4 Nf5 Nc2 Nd7 Rcxa4 Rxa4 Rxa4 Nb6 Nxd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 Bg5 h6 Bh4 g5 e3 Nf6 Bg3 d6 h4 Rg8 hxg5 hxg5 Nc3 e6 f3 Qe7 Qd2 Nc6 O-O-O Bd7 e4 fxe4 fxe4 O-O-O d5 exd5 exd5 Ne5 Re1 Kb8 Kb1 Bg7 a3 Rh8 Rxh8 Rxh8 Bxe5 dxe5 Nf3 e4 Qxg5 Rh5 Qd2 Qd6 Nd4 a6 Bc4 Rh4 Bb3 Rg4 Re2 e3 Qxe3 Nxd5 Nxd5 Bxd4 Qe7 Qxe7 Rxe7 Bc6 Nxc7 Rxg2 Bd5 Rg1 Ka2 Rg7 Rxg7 Bxd5 Nxd5 Bxg7 c4 Kc8 a4 Kd7 a5 Kd6 b4 Bd4 Kb3 Bf2 Nf4 Be3 Nd3 Kc6 Ka4 Bg1 b5 axb5 cxb5 Kc7 Nb4 Bf2 b6 Kb8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 d5 cxd5 cxd5 a6 g4 h5 h3 b5 Bg5 Qa5 Nd1 b4 Ng3 Nh7 gxh5 Nxg5 Qxg5 Bf6 Qh6 Bg7 Qg5 Bf6 Qh6 Bg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 Nf3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7 O-O a6 e4 c5 d5 c4 Bc2 Qc7 Nd4 Nc5 b4 cxb3 axb3 b4 Na4 Ncxe4 Bxe4 Nxe4 dxe6 Bd6 exf7 Qxf7 f3 Qh5 g3 O-O fxe4 Qh3 Nf3 Bxg3 Nc5 Rxf3 Rxf3 Qxh2 Kf1 Bc6 Bg5 Bb5 Nd3 Re8 Ra2 Qh1 Ke2 Rxe4 Kd2 Qg2 Kc1 Qxa2 Rxg3 Qa1 Kc2 Qc3 Kb1 Rd4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Qb6 a3 Nc6 Nxc6 bxc6 b3 Bb7 Bb2 d5 Bd3 c5 exd5 exd5 O-O-O O-O-O Na4 Qc7 Bf5 Kb8 Be5 Bd6 Qc3 d4 Bxd6 Qxd6 Qxc5 Qxf4 Kb1 Rd5 Rdf1 Qe5 Qc4 Rb5 Qxf7 Bd5 Qxg7 Rg8 Qh6 Bxb3 cxb3 Rxb3 Kc1 Qc7 Bc2 d3 Qf4 Rc8 Qxc7 Rxc7 Rf2 Ne4 Rf8 Ka7 Rf7 Rbb7 Rxc7 Rxc7 Rd1 Rxc2 Kb1 Rxg2 Rxd3 Rxh2 Rd7 Kb8 Re7 Nd2 Kc1 Nb3 Kd1 h5 Re3 Nd4 Nc5 a5 Nb3 Nc6 Rc3 Kb7 Ke1 Kb6 Kf1 Rh4 Kg2 Nd4 Nxa5 Kxa5 Rc5 Kb6 Re5 Kc6 Kg3 Rh1 Kg2 Kd6 Ra5 Rh4 Kg3 Rg4 Kh3 Ne2 Rxh5 Rg3 Kh4 Rxa3 Kg4 Ke6 Rb5 Rg3 Kh4 Rg1 Rg5 Rf1 Ra5 Kf6 Ra8 Rg1 Rf8 Ke5 Re8 Kf4 Rf8 Ke4 Re8 Kf3 Kh5 Ng3 Kh6 Nf5 Kh7 Kf4 Rb8 Rg7 Kh8 Rd7 Re8 Kg5 Re6 Nd4 Re1 Kf6 Rd1 Rd5 Ra1 Ne6 Ra6 Kf7 Ra7 Kg6 Ra8 Rd7 Rb8 Rc7 Kg8 Rc5 Ra8 Rb5 Kh8 Rb7 Rc8 Nc7 Rg8 Kh6 Rg1 Rb8 Rg8 Ne8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Na5 Bg1 Bf8 Qe1 Rb8 h3 Nd7 Bh2 Nc6 Rd1 Nxd4 Rxd4 b5 axb5 axb5 e5 f5 exd6 Bxd6 Nd5 Qa7 Rd1 b4 Bg1 Qa5 Bh5 g6 Qg3 Kf7 Bf3 Bf8 Rfe1 Bg7 Ne3 Kf8 Nc4 Qc7 Nd6 Re7 Bd4 Bxd4 Rxd4 Nf6 Red1 Ba6 Qh4 Nd7 Qe1 e5 fxe5 Nxe5 Qg3 Nxf3 gxf3 Ra8 Qf4 Be2 R1d2 Ra1 Kg2 Bf1 Kg3 Ba6 Rd1 Rxd1 Rxd1 Qxc2 Qd4 Kg8 Qd5 Kg7 Qd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nc6 Nc3 e5 e3 Nf6 d4 cxd4 exd4 e4 Ng5 Bb4 d5 Ne5 Qb3 a5 Bd2 O-O O-O-O d6 Ngxe4 Nxe4 Nxe4 Bf5 Ng3 Bd7 Kb1 Qe8 Bxb4 Ba4 Qe3 axb4 b3 Bd7 Be2 Qc8 Rd2 Re8 Qb6 Bf5 Ka1 Nd7 Qxb4 Nc5 Kb2 Bg6 Qc3 Qc7 f4 Qb6 Kc1 Be4 Bd1 Nd3 Rxd3 Bxd3 Qxd3 Rxa2 Bc2 g6 Ne4 Ra1 Bb1 Rxe4 Qxe4 Qxb3 Qc2 Qe3 Kd1 Qxf4 Re1 Qd4 Ke2 Qf6 Qd2 Qe5 Kf1 Rxb1 Rxb1 Qf5 Ke2 Qxb1 Qf4 Qc2 Kf1 Qb1 Ke2 Qb4 Kf3 Qc3 Ke2 Qe5 Qxe5 dxe5 c5 Kf8 Kd3 f5 Kc4 e4 Kb5 e3 c6 bxc6 dxc6 Ke7 Kb6 e2 c7 e1=Q c8=Q Qe6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 c5 d5 b5 Bg5 d6 Bxf6 exf6 e4 a6 a4 b4 Bd3 g6 Nbd2 Bg7 O-O O-O Nc4 a5 Nfd2 Qc7 f4 Nd7 Qe1 Ba6 Qh4 Rfe8 f5 Rad8 Rf3 Bc8 Raf1 Nf8 Ne3 Bd7 b3 h5 Ndc4 Bc8 g4 Bh6 gxh5 Bxe3 Nxe3 g5 Qf2 Kh8 Ng4 Nd7 Rg3 Re7 h4 c4 hxg5 cxd3 gxf6 Rxe4 Qd2 Qc5 Nf2 Re3 Qxe3 Qxe3 Rxe3 dxc2 Rc1 Nxf6 Rxc2 Nxd5 Rf3 Bb7 Rd2 Nc3 Rg3 Re8 Kh2 d5 Rg5 Ba6 h6 Ne4 Nxe4 dxe4 h7 e3 Rdg2 Re4 Rg8 Kxh7 R2g7 Kh6 f6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 b6 e3 Ne4 Qc2 Bb7 Bd3 Bxc3 bxc3 f5 O-O O-O Ne1 c5 f3 Nd6 Ba3 Na6 Qe2 Qe7 e4 fxe4 fxe4 Rxf1 Kxf1 e5 d5 Nc7 Kg1 Rf8 Nf3 Nf7 Bc1 d6 Rb1 Bc8 Qc2 Bd7 Be3 Nh8 a4 Ng6 a5 bxa5 Rb7 Bg4 Nd2 Ra8 Qa4 Bc8 Rb1 Qd8 Qxa5 Ne6 Qxd8 Nxd8 Ra1 a5 g3 h6 Kf2 Nf8 Nb1 Nd7 Na3 Nf6 Ke1 Nf7 Rb1 Bg4 Kf2 a4 Kg2 Bh5 h3 Bg6 Kf3 Rf8 Bd2 Nxe4 Ke3 Nfg5 Rf1 Rxf1 Bxf1 Nxg3 Bg2 Be4 h4 Bxg2 hxg5 hxg5 Nb5 Nf5 Kf2 Be4 Bxg5 Bd3 Na3 Kf7 Ke1 Kg6 Bd8 e4 Kd2 Kh5 Bc7 g5 Nb5 Bxc4 Nxd6 e3 Ke1 Nxd6 Bxd6 a3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 c3 d5 exd5 exd5 d4 a6 Bf4 Nc6 Be2 Nf6 O-O Be7 dxc5 Bxc5 Nbd2 O-O Nb3 Bb6 Nfd4 Ne4 f3 Nd6 Kh1 Qf6 Qc1 Re8 Bd3 h6 Be3 Bd7 Bg1 Ne5 Bc2 Ndc4 Nd2 Rac8 Re1 Nd6 Nf1 Nec4 Rxe8 Rxe8 b3 Na5 Qd2 Bc7 Re1 Rxe1 Qxe1 Nc6 Qd1 g6 Ne3 Ne7 Qe1 h5 Bf2 Nb5 Bh4 g5 Nxb5 axb5 Bf2 Be6 Nf1 Nc6 Qe3 h4 Qd3 h3 Ng3 hxg2 Kxg2 Ne5 Qh7 Kf8 Bf5 Nxf3 Bxe6 Nh4 Kf1 Qxe6 Bc5 Ke8 Qg8 Kd7 Qxg5 Qh3 Ke2 Qxh2 Bf2 Bxg3 Qxg3 Qxg3 Bxg3 Nf5 Bh2 Ne7 Kd3 Nc6 c4 Nb4 Kd4 bxc4 bxc4 Kc6 cxd5 Nxd5 Ke5 b5 Kf5 Kc5 Bg3 Kc4 Bd6 Kd4 a3 Kd3 Bf8 Nb6 Bc5 Nd7 Be7 Kd4 Bb4 Kc4 Bd6 Kb3 Kg5 Nb6 Kf6 Nc4 Bf8 Nxa3 Kxf7 Nc2 Ke6 Kc4 Kd7 Nd4 Be7 Kd5 Kc7 Nc6 Kb6 Nd4 Bc5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 h4 gxh4 Rxh4 Nc6 Nb3 Be6 Qd2 Qb6 Nd5 Bxd5 exd5 Nce5 c3 Ng6 Rh3 h4 Bxg4 hxg3 Rxh8 Bxh8 Qe3 gxf2 Kxf2 Qxe3 Kxe3 Ne5 Be2 Kd7 Nd2 Rg8 Rh1 Bf6 Kf2 Ng4 Bxg4 Rxg4 Rh7 Ke8 Kf3 Rg5 Ne4 Rf5 Ke3 Rxd5 Nxf6 exf6 Rh8 Kd7 Rb8 Re5 Kf3 Kc7 Rf8 Re7 g4 Kd7 Kf4 Ke6 c4 Rc7 b3 b5 Re8 Kd7 Ra8 bxc4 bxc4 Rxc4 Kf5 Ra4 Kxf6 Rxg4 Ra7 Kc6 Rxa6 Kc5 Kxf7 d5 Ra5 Kd6 Kf6 Rf4 Kg5 Rf1 a4 Ke5 Ra8 Rg1 Kh6 d4 a5 d3 Rd8 Ke4 a6 Ra1 Rd6 Ke3 Re6 Kf2 Rd6 Ke2 Re6 Kd1 Kg5 d2 Kf4 Kc2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 c3 Nf6 e5 Nd5 d4 cxd4 cxd4 d6 Bc4 Nb6 Bb3 Nc6 exd6 Bxd6 O-O O-O Nc3 Nd5 Re1 b6 Nxd5 exd5 h3 Bf5 Bg5 Qd7 Bxd5 Bh2 Kxh2 Qxd5 Qb3 Qd6 Kg1 h6 Bh4 Be6 Qd3 Nb4 Qd2 Rfc8 Bg3 Qf8 d5 Bxd5 a3 Bxf3 Qxb4 Qxb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Qf3 Qb6 Nb3 Qc7 g4 b5 g5 Nfd7 Be3 Nb6 O-O-O N8d7 Qh3 b4 Ne2 Nc4 Kb1 Bb7 Ned4 Nxe3 Qxe3 g6 Bh3 Qb6 Rhf1 O-O-O c3 Kb8 cxb4 Qxb4 Rc1 Nc5 Rfd1 Be7 e5 Nxb3 axb3 Rc8 Bf1 Qb6 b4 Rhd8 b5 axb5 Bxb5 dxe5 Qxe5 Bd6 Qe3 Rxc1 Rxc1 Bc7 Nc6 Bxc6 Qxb6 Bxb6 Rxc6 Be3 Rc4 Rd5 Re4 Bd2 Be8 Rf5 Rd4 Bxf4 Rd8 Kc7 Rd7 Kc8 Rxf7 Kd8 Rxf5 exf5 Bf7 Ke7 Bg8 Kf8 Bd5 Bxh2 Kc2 Bf4 Kd3 Bxg5 b4 Bd8 Ke3 Kg7 Kf4 Kf6 b5 h5 Kg3 g5 Kf3 g4 Kg2 h4 Kf2 Bb6 Kg2 h3 Kg3 Kg5 Kh2 Bc7 Kg1 g3
Result: 0-1
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 Bb4 Bg2 O-O O-O Re8 d3 Bxc3 bxc3 e4 dxe4 Nxe4 Qc2 d6 Nd4 Qe7 Rd1 Bd7 f3 Nc5 e4 Na5 Qe2 f5 Ba3 fxe4 fxe4 Rad8 Rf1 Nc6 Rae1 b6 Bc1 Ne5 Nf5 Qe6 Bg5 Rc8 Qh5 Qg6 Qh4 Be6 Ne7 Rxe7 Bxe7 Bxc4 Rf5 Re8 Bg5 Be6 Rff1 Bc4 Rf5 Be6 Ref1 h6 Be3 Bxf5 exf5 Qg4 Bd5 Kh7 Qxg4 Nxg4 Bd4 Nf6 Bf3 Nd3 h4 Ne5 Bd1 c5 Be3 Nd5 Bd2 Nc4 Re1 Rxe1 Bxe1 Nde3 Be2 d5 Kf2 Nxf5 Bd3 g6 h5 Ne5 hxg6 Kxg6 Bc2 h5 Ke2 Kf6 Bf2 d4 Bxf5 Kxf5 cxd4 cxd4 Bxd4 Nc6 Bf2 Ke4 a4 Nd4 Kd2 Nf5 Ke2 Nd6 a5 bxa5 Bxa7 a4 Bc5 Nf5 Kf2 Kd3 Kf3 Nd4 Kg2 Ke4 Kh3 Ne6 Ba3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7 Bf2 Nc6 Qd2 Ne5 O-O g4 f4 Nc4 Qe2 Rc8 b3 Na3 Nd5 e6 Nb4 Qa5 Qe1 h4 Be3 h3 g3 Nb5 Rd1 Nc3 Nd3 Qc7 Rc1 Nxe4 f5 e5 f6 Nxf6 Nf5 Bxf5 Rxf5 Qc6 Qe2 Qe4 Rf2 Nd5 Re1 Qxe3 Qxg4 O-O Rxe3 Nxe3 Qxh3 Nxc2 Qd7 Nd4 Qxb7 a5 Kg2 Rc3 Nb2 Nc2 Nc4 d5 Nd6 Ne3 Kh3 f5 Qd7 f4 Qe6 Kh7 Nf7 Rxf7 Qxf7 Rc6 gxf4 Rf6 Qc7 e4 f5 d4 Qe7 Rh6 Kg3 Nd1 Rf4 e3 Rg4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7 Bf2 Nc6 Qd2 Ne5 O-O e6 b3 Ng6 Rad1 g4 f4 h4 Nde2 h3 g3 Rc8 e5 d5 Nd4 O-O Qd3 f5 Nce2 Rf7 Rc1 Bf8 c3 Ne7 Rfd1 Nc6 c4 dxc4 Rxc4 Qe8 Qd2 b5 Rcc1 Nb4 Nc3 Rc7 a3 Qa8 Be3 Nd3 Rc2 b4 Na2 Nc5 Qe2 bxa3 b4 Ba4 bxc5 Bxc2 Nxc2 Bxc5 Bxc5 Rxc5 Ne1 Rb5 Qc4 Rb2 Qxe6 Qa7 Kh1 Rxh2 Kxh2 Qf2 Kh1 Qf1 Kh2 Qe2 Kg1 Qxd1 Kf1 Qd7 Qxa6 h2 Qg6 Rg7 Qh5 Rh7 Qg6 Qg7 Qe8
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 Be3 Ng4 Bg5 f6 Bc1 Nc6 O-O f5 Bg5 Bf6 Bxf6 Nxf6 exf5 Bxf5 d5 Ne7 Qd2 Kg7 Nh4 Bd7 f4 exf4 Qxf4 Nf5 Nxf5 Bxf5 g4 Bd7 g5 Ng4 Qxf8 Qxf8 Rxf8 Rxf8 Bxg4 Bxg4 Nb5 Rf4 Nxc7 Rxc4 Ne8 Kf8 Nf6 Bf5 Re1 Rc2 Re8 Kf7 Rh8 Rxb2 Rxh7 Kf8 h4 a5 h5 gxh5 Nd7 Bxd7 Rxd7 Rxa2 Rxd6 Rd2 Rd8 Kg7 d6 b5 Kf1 a4 Ke1 Rd5 d7 b4 Rb8 Rxd7 Rxb4 Ra7
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Bxc6 dxc6 d3 Nd7 b3 O-O Bb2 f6 d4 exd4 Nxd4 Nc5 Nd2 Ne6 Nf5 Bc5 Nc4 Qxd1 Raxd1 Ng5 Ng3 Bg4 Rd3 Rad8 h3 Rxd3 cxd3 Bc8 Ba3 Bxa3 Nxa3 Rd8 Rd1 Ne6 Nc2 c5 Ne2 g5 f3 Kf7 Kf2 b6 Ke3 a5 g3 Ba6 f4 gxf4 gxf4 c4 bxc4 Bxc4 Nc3 Ba6 f5 Nc5 d4 Nd7 Kf4 Re8 Ne3 c5 Ng4 h5 Nf2 cxd4 Rxd4 Ne5 Rd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Be7 h3 Nc6 a3 Ne4 c4 Bf5 O-O dxc4 Bxc4 O-O d5 Na5 Ba2 c5 Re1 c4 Nbd2 Nxf2 Kxf2 Bc5 Re3 Bxe3 Kxe3 Re8 Kf2 Qxd5 Kg1 Rad8 Kh1 b5 Qf1 Bd3 Qg1 Nc6 Nb1 Nd4 Nxd4 Qxd4 Nc3 Qxg1 Kxg1 Re1 Kf2 Rde8 Nxb5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 g3 Bb4 Bd2 Be7 Bg2 d5 Nf3 O-O O-O c6 Qc2 b6 Ne5 Nfd7 Nd3 Bb7 Rd1 a5 a3 Ba6 cxd5 cxd5 Be3 Ra7 Nc3 Rc7 Bf4 Rc8 Bxb8 Nxb8 Rac1 Nc6 e3 Bc4 Bf1 Qd7 Nf4 b5 Bxc4 bxc4 e4 Bf6 exd5 Nxd4 Qe4 e5 Nh5 Bg5 f4 f5 Qg2 Qf7 Nxg7 Qxg7 fxg5 Qxg5 Kh1 e4 g4 Kh8 gxf5 Qxg2 Kxg2 Nxf5 Kh1 Nd6 Re1 Rf4 a4 Rb8 Re2 Kg7 Rce1 Rb4 Nb5 Nxb5 axb5 Kf7 d6 Ke6 Rd2 Kd7 Rg1 Rf7 Re1 a4 Re3 Rg7 Rxe4 a3 Re7 Rxe7 dxe7 Kxe7 bxa3 Rb1 Kg2 c3 Re2 Kd6 Kf3 Kd5 a4 Kd4 a5 Rxb5 a6 Ra5 Re4 Kd5 Re3 c2 Rc3 Rc5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Nxd7 O-O Ngf6 Qe2 e6 b3 g6 Bb2 Bg7 d4 cxd4 Bxd4 O-O Rd1 b5 Qxb5 Nxe4 Qe2 d5 c4 Rc8 Bxg7 Kxg7 Qb2 Qf6 Qxf6 Ndxf6 Nbd2 Rfd8 Nxe4 Nxe4 cxd5 Nc3 Rd2 Rxd5 Kf1 Rdc5 Re1 Nd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 g3 d5 Bg2 Be7 O-O O-O c4 dxc4 Qc2 a6 a4 Bd7 Qxc4 Bc6 Bf4 a5 Nc3 Na6 Rac1 Nb4 Rfe1 Bd6 Bg5 Bd5 Nxd5 exd5 Qb3 h6 Bxf6 Qxf6 e3 Rae8 Qd1 Re7 Rc3 Rfe8 Qb1 c6 Rd1 Qe6 Nd2 Na6 Nb3 Bb4 Rcc1 Qd6 Qc2 Nc7 Nc5 Ne6 Nd3 Qd8 Nxb4 axb4 h4 Qa5 b3 Nf8 Qc5 Qxc5 dxc5 Ne6 Bh3 g6 Bxe6 Rxe6 Rd4 Re4 Rcd1 g5 hxg5 hxg5 Kf1 Kg7 Ke2 g4 Rxe4 Rxe4 a5 Re8 Ra1 d4 Ra4 dxe3 Rxb4 exf2 Kxf2 Ra8 Rxg4 Kf6 b4 Rd8 Re4 Rd1 Re8 Rb1 Rb8 Rxb4 a6 Ra4 axb7 Rb4 g4 Ke7 g5 Kd7 Rf8 Rxb7 Rxf7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 b6 a3 Bb7 Nc3 g6 d5 Bg7 g3 exd5 cxd5 O-O Bg2 c6 d6 c5 O-O Re8 Re1 Ne4 Nd5 Bxd5 Qxd5 Nc6 Nh4 Nf6 Qd1 Re6 Bh3 Re4 Nf3 h6 Bd2 Re8 Bc3 Ne4 Bxg7 Kxg7 Qc2 f5 Rad1 Qf6 Rd5 Re6 Red1 Rxd6 Rxd6 Nxd6 b4 cxb4 axb4 Rd8 Bg2 Nb5 Qd3 a6 Qc4 Qc3 Rc1 Qxc4 Rxc4 Nd6 Rc1 Nxb4 Nd4 Rc8 Rd1 Kf6 h4 Ne4 g4 d5 gxf5 gxf5 Nxf5 Nc3 Rd2 a5 Nxh6 a4 Ng4 Ke7 Ne3 a3 Nf5 Kf6 Nd4 a2 Nb3 Nb1 Rd4 Rc4 Rd1 Rxh4 e3 Rc4 Bf1 Rc3 Na1 Nc2 Nxc2 Rxc2 Bd3 Rb2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 c5 d5 a6 a4 h6 Be3 e6 Qd2 exd5 cxd5 Re8 Nge2 Nbd7 Bxh6 Nxe4 Nxe4 Qh4 g3 Qxh6 Qxh6 Bxh6 Nxd6 Re3 f4 Nb6 Kd2 Nxd5 Nc3 Re6 Nxc8 Nxc3 bxc3 Rxc8 Bc4 Rd8 Kc2 Re4 Kb3 b5 axb5 axb5 Bxb5 c4 Kb4 Bf8 Ka5 Ra8 Ba6 Bg7 Rhc1 Bf6 Kb5 Rb8 Kc5 Re7 Kxc4 Ra7 Ra3 Rb2 Rca1 Rc7 Kd3 Rxh2 Re1 Ra7 Re3 Rb2 Ra4 Rd7 Ke4 Bd8 Re2 Rb6 Bc4 Kg7 g4 Bc7 Ra6 Rb1 f5 Rf1 f6 Kh6 Rh2 Bxh2 Bxf1 Kg5 Be2 Bg1 Rc6 Rd8 Bc4 Rd7 Be2 Bh2 Kf3 Be5 Rc5 Kxf6 Ke4 Bd6 Rc6 Ke7 c4 Bh2 Ra6 Bg1 g5 Rc7 Ra5 Rc6 Kf3 Re6 Ra3 Re5 Rb3 Rxg5 Rb7 Kf8 Ke4 Bc5 Rc7 Be7 Bf3 Ke8 Ra7 Rc5 Kd4 Rf5 Bc6 Kf8 Ra8 Kg7 Re8 Kf6 Bd5 Rf4 Kd3 g5 Rc8 g4 c5 g3 c6 Rf5 Kc4 Rf2 Rg8 Rc2 Kd3 Rc5 Ke4 Bd6
Result: 0-1
Processing game...
FEN: 
Moves: c4 c6 e4 d5 exd5 cxd5 d4 Nf6 Nc3 Nc6 Bg5 Be6 a3 Bg4 f3 Be6 c5 g6 Bb5 Bg7 Nge2 O-O O-O Bf5 b4 a6 Ba4 h6 Be3 Ne8 Qd2 Kh7 Ng3 e6 Nge2 g5 Bc2 Bxc2 Qxc2 f5 Rab1 Nf6 a4 Qe8 Bf2 Qg6 b5 Na5 bxa6 Rxa6 Rb5 Nd7 Rfb1 Nb8 Na2 Nbc6 Nb4 Ra7 Nd3 f4 Qd1 Rf7 Nb2 Ra8 h3 h5 Qd3 Ra7 h4 g4 Qxg6 Kxg6 Nd3 Bh6 Be1 Nc4 Ra1 gxf3 gxf3 Kf5 a5 Rg7 Kh1 Ra8 Bc3 Ne3 Rg1 Rxg1 Nxg1 Ra7 Ne2 Nc4 Kg2 N4xa5 Bxa5 Nxa5 Ne5 Nc6 Nxc6 bxc6 Rb6 Rc7 Kf2 Bg7 Ke1 Bf6 Kd2 Bxh4 Rb8 Bf6 Rf8 h4 Ng1 Kg6 Kd3 Ra7 Nh3 Ra3 Ke2 Kf5 Nxf4 Kxf4 Rxf6 Kg5 Rxe6 h3 Kf2 Ra2 Kg1 h2 Kh1 Kf4 Rf6 Kg3 Rg6 Kh3 Rh6 Kg3 Rg6 Kxf3 Rxc6 Kg3 Rg6 Kf3 c6 Rc2 Rh6 Ke4 Rh4 Kd3 Rxh2 Rxc6 Rh4 Rc2 Kg1 Re2 Kf1 Re4 Rh2 Kxd4 Rd2 Kc4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 c5 d5 a6 a4 h6 Be3 e6 Qd2 exd5 cxd5 Re8 Nge2 Nbd7 Nd1 Ne5 Nec3 g5 Be2 Nh5 O-O Qf6 a5 Nf4 Na4 Bd7 Ndc3 Bxa4 Rxa4 Rac8 Kh1 c4 Rb4 Nxe2 Qxe2 Qe7 Rb6 Nd3 Qd2 f5 Bg1 f4 b3 Ne5 Rb4 cxb3 Rxb3 Nc4 Qa2 Qc7 e5 Bxe5 Ne4 Nxa5 Rbb1 Nc4 Rfc1 b5 Qxa6 Qf7 Qxb5 Na3 Qa6 Ra8 Qd3 Nxb1 Rxb1 Re7 h3 Qg6 Kh2 Rf8 Rb6 h5 Qa6 g4 Nxd6 gxf3 Qf1 Bxd6 gxf3 Rg7 Bf2 Qf6 h4 Rg6 Qh3 Qf5 Rc6
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 O-O Be7 Bb3 O-O f4 b5 e5 dxe5 fxe5 Nfd7 Qh5 Bc5 Be3 Bxd4 Bxd4 Nc6 Be3 Ncxe5 Rad1 Bb7 Rd4 Ng6 Ne2 Qe7 Nf4 e5 Nxg6 hxg6 Qxg6 exd4 Bxd4 Ne5 Qg3 Rae8 Rf5 Nf3 gxf3 Qe1 Qxe1 Rxe1 Kf2 Rd1 Bc5 Bc8 Rf4 g5 Re4 Rfd8 Re5 R8d2 Kg3 Be6 Rxg5 Kh7 c4 Rh1 cxb5 axb5 Be3 Rxb2 Rh5 Kg6 Rg5 Kh7 Rh5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 f4 Be7 O-O O-O Be3 Qc7 g4 Re8 g5 Nfd7 Bd3 Nc6 Qh5 g6 Qh4 Bf8 Rf3 Bg7 Nde2 b5 Rh3 Nf8 f5 Ne5 f6 Bh8 a3 Rb8 b4 Bb7 Rf1 Rbc8 Bd4 Nc6 Be3 Ne5 Rf4 Qd7 Qh6 Nxd3 cxd3 e5 Rf1 Rc7 Bb6 Rc6 Nd5 Bc8 Re3 Qb7 Bd8 Ne6 Ne7 Rxe7 fxe7 Qd7 Rh3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Re1 Nf6 e5 Nd5 Nc3 Nc7 Bxc6 dxc6 Ne4 Ne6 d3 O-O Be3 b6 Qd2 f5 exf6 exf6 Bh6 a5 Bxg7 Kxg7 Re2 Ra7 Rae1 Raf7 h4 h5 b3 Qc7 Ng3 Nd4 Nxd4 cxd4 c3 Rd8 Re4 dxc3 Qxc3 c5 Ne2 Bf5 Nf4 Qd7 Ne6 Bxe6 Rxe6 Qxd3 Qxd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 Nc3 Nf6 e3 e6 d4 d5 cxd5 exd5 Bb5 cxd4 Nxd4 Bd7 O-O Bd6 Nf3 a6 Be2 Be6 a3 O-O b4 Qe7 Bb2 Rfe8 Nd4 Be5 Nxc6 bxc6 Na4 Bxb2 Nxb2 c5 bxc5 Qxc5 Nd3 Qd6 Nf4 Bf5 Bf3 Rad8 Qd4 Be4 Be2 Bf5 Bf3 Be4 Be2 Bf5 Qb4 d4 Qxd6 Rxd6 Rfd1 g5 Nh5 Nxh5 Bxh5 d3 Rd2 g4 f3 gxf3 Bxf3 Rxe3 Rc1 Re5 Kf2 Kg7 Rc4 Ra5 Rc3 Rb5 h4 h6 Ke3 a5 Kf2 Rd4 g3 Re5 Rc7 Ra4 Rc3 Bg6 Rb3 Rc4 g4 Rc2 Rb2 Rxd2 Rxd2 f6 Kg3 Rc5 Kf4 Rc3 Rb2 Rxa3 Rb7 Kf8 Rd7 Ra4 Ke3 f5 h5 Be8 Rxd3 fxg4 Be4 Bxh5 Rd6 Kg7 Rd7 Kf6 Rd6 Kg5 Bc2 Ra3 Kf2 g3 Kg1 Bf3
Result: 0-1
Processing game...
FEN: 
Moves: e4 g6 d4 Bg7 Nf3 d6 Be2 Nf6 Nc3 O-O O-O c6 h3 Nbd7 Bf4 Qa5 Qd2 e5 dxe5 dxe5 Bh6 Re8 Bxg7 Kxg7 a3 Qc7 Nh2 Nf8 Ng4 Nxg4 Bxg4 Bxg4 hxg4 Ne6 Ne2 Rad8 Qc3 Nd4 Rfe1 h6 Nxd4 exd4 Qf3 Qe5 Rad1 c5 b3 Re6 g3 b5 Qd3 a6 Kg2 Qc7 c4 Qc6 Rc1 Rde8 cxb5 axb5 b4 c4 Qxd4 Kg8 f3 Rd6 Qe3 Red8 Re2 Rd3 Qxh6 Qf6 Rf2 Rxa3 Rh1 Qg7 Qg5 Rad3 Qxb5 c3 Rc1 Rd2 Rc2 Rxc2 Rxc2 Rd2 Rxd2 cxd2 Qe8 Qf8 Qd7 Qxb4 Kf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 d6 Bg5 e6 Qd2 a6 O-O-O h6 Be3 Be7 f3 Nxd4 Bxd4 b5 Kb1 Rb8 Qe3 b4 Na4 Qc7 b3 O-O Bd3 e5 Bb2 Bd7 Bxa6 Ra8 Qe2 Bxa4 bxa4 Nd7 Qc4 Nc5 Bb5 Qa5 c3 bxc3 Bxc3 Qa7 Rc1 Nxa4 Qxa4 Qxa4 Bxa4 Rxa4 Rc2 f5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 h3 Bh5 O-O Nc6 Be3 cxd4 cxd4 Bb4 a3 Ba5 Nc3 Qd6 Nb5 Qe7 Ne5 Bxe2 Qxe2 O-O Rac1 Rac8 Bg5 Bb6 Bxf6 gxf6 Nc4 Rfd8 Nxb6 axb6 Rfd1 f5 Qe3 Qf6 d5 Rxd5 Rxd5 exd5 b3 Kh8 Qxb6 Rg8 Qc5 d4 Nd6 f4 Nxb7 Ne5 Qd5 f3 g3 Nd3 Rc7 Re8 Nd6 Re1 Kh2 Nxf2 Nxf7 Kg7 Ng5 Kh6 Rxh7
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 e6 g3 c5 Bg2 Nc6 O-O Nf6 c4 dxc4 Ne5 Bd7 Na3 cxd4 Naxc4 Bc5 Qb3 O-O Qxb7 Nxe5 Nxe5 Rb8 Qf3 Bd6 Nc6 Bxc6 Qxc6 e5 Rb1 Rb6 Qa4 Qb8 Bg5 Be7 b4 Bxb4 Bxf6 gxf6 Qd7 Qc8 Qxa7 Rb8 Qa4 Bc3 Rxb8 Qxb8 Be4 Qc7 Qa6 Kg7 Qd3 Rb8 Bxh7 Rb2 Be4 Rxa2 h4 Qc8 Qf3 Ra1 Rxa1 Bxa1 Qh5 Qh8 Qg4 Kf8 Qc8 Kg7 Qg4 Kf8 Bd5 Ke7 Bc6 Kf8 Bd5 Ke7 Qf3 Bc3 Bc4 Qc8 Qd5 Qe6 Qb5 Qd7 Qc5 Qd6 Qa7 Qd7 Qa8 Qc7 Qa3 Qd6 Qa2 f5 Bxf7 e4 Bh5 Qf6 Qa3 Kd7 Qa7 Kd8 Qb8 Kd7 Be8 Ke7 Bb5 Bd2 Qc7 Kf8 Bc4 Bc3 Kg2 Be1 Kf1 Bc3 f4 exf3 exf3 Bd2 f4 Ke8 Qc8 Ke7 Qc5 Kd8 Bd3 Be3 Qxf5 Qc6 Qf8 Kc7 Qe7 Kc8 Bf5 Kb8 Qd8 Kb7 Qd7 Qxd7 Bxd7 Kc7 Bb5 Kd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 O-O Nc6 Be3 cxd4 cxd4 Bb4 a3 Ba5 Nc3 Qd6 Ne5 Bxe2 Qxe2 Bxc3 bxc3 Nxe5 Bf4 Nf3 Qxf3 Qd5 Qd3 Rc8 Rfc1 Qc4 Qxc4 Rxc4 Rcb1 b6 Bb8 Ra4 Rb4 Ra5 Rc4 O-O Bd6 Ra8 Rc6 b5 Kf1 Ra4 Rb1 a6 Ke2 h5 Kd3 Rd8 Be7 Rd7 Bxf6 gxf6 Rb3 Kg7 Ke3 e5 g3 exd4 cxd4 Re7 Kf3 Rd7 Rd3 Raxd4 Rxd4 Rxd4 Rxa6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 c6 c4 e6 Nbd2 Nf6 e3 Nbd7 Bd3 Bd6 e4 dxe4 Nxe4 Nxe4 Bxe4 O-O O-O h6 Bc2 e5 Re1 exd4 Qxd4 Bc5 Qc3 a5 a3 Nf6 Be3 Bxe3 Rxe3 Bg4 Ne5 Re8 Rae1 Be6 f4 Qc8 h3 b5 f5 Bxc4 Nxc4 bxc4 Rxe8 Nxe8 Re4 Nf6 Rxc4 Nd5 Qe5 Qd7 Rg4 f6 Qd4 Kh7 Re4 Rd8 Kh1 Qc7 Qf2 Qb8 Ba4 c5 Bc6 c4 Rxc4 Nb4 Bf3 Nd3 Qh4 Qxb2 Qg3 Qxa3 Rc7 Qf8 Ra7 Ne5 Rxa5 Qf7 Rxe5 fxe5 Qxe5 Re8 Qf4 Qf6 Bh5 Rf8 Bg6 Kh8 Qc7 Qd4 Kh2 Ra8 Bh5 Qf6 Bg6 Rg8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nc3 Nc6 d4 exd4 Nxd4 Bb4 Nxc6 bxc6 Bd3 d5 exd5 cxd5 O-O O-O Bg5 c6 Qf3 Be7 Rae1 Re8 Ne2 h6 Bf4 Bd6 Nd4 Bg4 Qg3 Bxf4 Qxf4 Qb6 c4 Bd7 cxd5 cxd5 Rxe8 Rxe8 Qd2 Ne4 Bxe4 dxe4 b3 Rd8 Qc3 f5 Rd1 Be6 Qe3 Bf7 Qc3 f4 Rd2 Qf6 g3 Rd5 a3 Kh7 Kg2 Qe5 f3 e3 Rd3 e2 gxf4 e1=Q fxe5 Qxc3 Rxc3 Rxd4 b4 Bc4 Kf2 g5 Re3 Rd2 Ke1 Rd3 Kf2 Kg6 Rxd3 Bxd3 Ke3 Bc2 Kd4 Kf5 Kd5 h5
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 d4 c6 c4 e6 Nbd2 Nf6 e3 c5 b3 Nc6 Bb2 cxd4 exd4 Be7 Rc1 O-O Bd3 Bd7 O-O Nh5 Re1 Nf4 Bb1 Bd6 g3 Ng6 Ne5 Rc8 Nxd7 Qxd7 Nf3 Bb4 Re3 Rfd8 h4 Nge7 a3 Ba5 b4 Bc7 c5 Re8 Qd3 g6 Re2 Nf5 Bc3 h5 b5 Nce7 Bd2 Kg7 a4 Ra8 a5 a6 b6 Bb8 Bc2 Nc6 Ba4 Re7 Bc3 Ne5 dxe5 Qxa4 Nd4 Nxd4 Qxd4 Qd7 Bd2 Re8 Bg5 Rc8 Bf6 Kh7 c6 bxc6 Qc5 Kh6 Rb2 Qb7 Rb4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Bd3 Nf6 O-O Ne5 h3 Bc5 Kh1 d6 f4 Ng6 Qe1 O-O f5 Ne5 Qh4 Bd7 g4 Qb6 g5 Nh5 Qxh5 Bxd4 Bxd4 Qxd4 f6 Bc6 fxg7 Kxg7 Qh6 Kg8 Qf6 Qe3 Kh2 Nd7 Qh6 Ne5 Qf6 Nd7 Qh6 Ne5 Qf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Be3 e5 Nf3 c6 O-O exd4 Nxd4 Re8 f3 d5 cxd5 Nxd5 Nxd5 cxd5 Qb3 Nc6 Rad1 Nxd4 Bxd4 Bxd4 Rxd4 Qf6 Qxd5 Be6 e5 Bxd5 exf6 Rxe2 Rxd5 Rxb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 d5 Nbd7 Be3 Ng4 Bg5 f6 Bh4 h5 h3 Nh6 Nd2 g5 Bg3 f5 Bxh5 f4 Bh2 Nf6 Bf3 c6 Bg1 cxd5 cxd5 Bd7 Be2 a6 f3 Nh5 Bf2 Ng3 Rg1 b5 Bd3 Qb8 Ne2 Nh5 b4 a5 a3 a4 Rc1 Bf6 Nf1 Bd8 Qd2 Nf7 Nc3 Qb7 Rc2 Rb8 Nh2 Ng3 Kd1 Kg7 Kc1 Bb6 Bxb6 Qxb6 Qf2 Qd4 Qxd4 exd4 Ne2 Ne5 Rd1 Nxd3 Rxd3 Nxe2 Rxe2 Rbc8 Rc2 Rxc2 Kxc2 Rc8 Kb2 Rc4 Nf1 Kf6 Nd2 Ke5 Nxc4 bxc4 Rd1 Bb5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 cxd5 exd5 Bg5 Bf5 e3 Nbd7 Bd3 Bxd3 Qxd3 Be7 O-O O-O Nd2 Re8 Rab1 a5 Qc2 Bd6 Bh4 Qb8 Bg3 Bxg3 hxg3 Qd6 Rfe1 Re6 a3 Rae8 Na4 h5 Nc5 Nxc5 dxc5 Qd8 b4 d4 e4 d3 Qc4 Ng4 Qc3 h4 f3 Ne5 g4 h3 Nc4 Nxc4 Qxc4 hxg2 Red1 Rh6 Kxg2 axb4 axb4 Qc7 Rh1 Rxh1 Rxh1 Rd8 Rd1 d2 Qc3 Qf4 Kf2 g6 Ke2 Qh2 Ke3 f5 gxf5 gxf5 e5 f4 Ke4 Qe2 Kf5 Rf8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 g6 Bxc6 dxc6 h3 Nf6 d3 Bg7 Nc3 O-O Be3 c4 O-O cxd3 cxd3 Qa5 d4 Be6 Qc2 h6 a3 Rfd8 b4 Qh5 Ne2 Qb5 Nf4 Qc4 Qb1 Qb3 e5 Ne8 Nxe6 Qxe6 b5 Rac8 bxc6 Qxc6 Qb4 Qd7 Rfd1 b6 d5 Nc7 d6 exd6 Rxd6 Qe8 Rad1 Ne6 Qh4 Qf8 a4 Rxd6 exd6 Rd8 d7 g5 Qe4 Qe7 Qb7 Nf8 Qxa7 Rxd7 Rxd7 Qxd7 Qxd7 Nxd7 Nd2 Be5 Nc4 Bc7 g4 f6 Kg2 Kf7 Kf3 Ke6 Ke4 h5 gxh5 f5 Kd3 f4 Bd4 Kf5 f3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 g3 b5 Bg2 Bb7 O-O c5 c3 Na6 Bg5 Be7 Nbd2 O-O a3 Rc8 Re1 cxd4 cxd4 h6 Bxf6 Bxf6 e3 d6 Qb1 Qb6 Qd3 Nb8 Rac1 Nd7 h4 Be7 Nh2 d5 Bf1 b4 a4 a6 Nhf3 Bd6 Qb3 Qa5 Bd3 Nb6 Bc2 Nc4 Qd3 f5 Rb1 Qc7 Rec1 Bxg3 fxg3 Qxg3 Kf1 f4 Qh7 Kf7 Ng5 hxg5 Nxc4 fxe3 Nxe3 Qxe3 Kg2 Qxd4 Qg6 Ke7 Qxg5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Nbd7 O-O O-O Re1 c5 e4 dxc4 Nxc4 Bb7 e5 Nd5 Bd2 Rc8 Qe2 b5 Na5 Ba8 Qxb5 cxd4 Nxd4 Rc5 Qa6 Nc7 Qe2 Bxg2 Kxg2 Nxe5 b4 Rd5 Bc3 Qa8 Kg1 Bf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5 f3 f4 Bf2 g5 a4 a5 Nd3 b6 b4 axb4 Nxb4 Nf6 Ra3 Bd7 Nb5 Kh8 Be1 Rg8 g4 fxg3 hxg3 g4 Kg2 gxf3 Bxf3 Bh6 Rh1 Bg5 Nd3 Ng6 Nf2 Qc8 Kf1 Rf8 Kg2 Rg8 Kf1 Rg7 Ke2 Rf7 Bd2 Bxd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7 O-O a6 e4 c5 d5 c4 Bc2 Qc7 dxe6 fxe6 Nd4 Nc5 Qe2 Bd6 f4 e5 Nf5 O-O Nxd6 Qxd6 fxe5 Qxe5 Rf5 Qc7 Bg5 Nfxe4 Rxf8 Rxf8 Nxe4 Qe5 Nf6 gxf6 Bxh7 Kxh7 Qh5 Kg8 Qg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 a6 O-O-O Nxd4 Qxd4 Be7 f4 b5 Bxf6 gxf6 e5 d5 Kb1 b4 Ne2 a5 Ng3 f5 Nh5 Rb8 g4 fxg4 f5 Rg8 Nf6 Bxf6 exf6 Qd6 Bg2 Rg5 Bxd5 Bd7 Rhe1 h6 fxe6 fxe6 Qa7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 c4 Nc6 Nc3 g6 d4 Bg7 d5 Bxc3 bxc3 Na5 O-O f6 Nd2 b6 Qe2 Qa4 f4 Nh6 e5 O-O-O Rb1 Nf5 g4 Nh4 exf6 exf6 Qf2 g5 Ne4 Qe8 Re1 Qg6 fxg5 Rhe8 Nxd6 Rxd6 Rxe8 Qxe8 Bf4 Nxc4 Bxd6 Nd2 Rd1 Qe4 Bg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 a4 Nc6 O-O Be7 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Rb8 g4 Nxd4 Bxd4 e5 fxe5 dxe5 Ba7 Ra8 g5 Rd8 Qe2 Ne8 Be3 Be6 Qf2 Qc4 Bb6 Rdc8 Be3 Bc5 Rad1 Bxe3 Qxe3 Qb4 Rd3 Rd8 b3 Rd4 Rfd1 Rc8 R1d2 Qc5 Kg2 b5 axb5 axb5 Nd1 Rxd3 Qxd3 f6 gxf6 Nxf6 Ne3 Kh8 c4 bxc4 Nxc4 h6 Rc2 Qb5 Rc1 Qc5 Rc2 Qb5 Rc1 Qb4 Qc3 Qe7 Qe3 Nd7 Be2 Rb8 Nd2 Nf8 Kh1 Ng6 Nc4 Bg8 Rf1 Qc7 Qg3 Bh7 Rf2 Nf4 Qe3 Bg8 Bf1 Rf8 Nd2 Qc1 Nc4 Qa1 Qc5 Ne6 Qe3 Nf4 Qc5 Re8 Qe3 Bxc4 bxc4 Rd8 Qf3 Qe1 Qg3 Rd1 h4 Rd3 Qg1 Qxe4 Kh2 Qe3
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 g3 g6 b3 Bg7 Bb2 O-O Bg2 Rfe8 O-O Nb6 Re1 d5 Qc2 Qc5 Nd2 Rad8 Rac1 d4 Nf3 d3 Qd2 Bc8 h3 h5 Rcd1 Bf5 e6 Rxe6 Rxe6 Bxe6 Bxg7 Kxg7 Qc3 Kg8 Rxd3 Rxd3 Qxd3 Nd7 Qc3 Bf5 Nd4 Qe5 Qd2 c5 Nxf5 Qxf5 Qa5 Ne5 Qxa7 h4 Qa8 Kg7 Qe4 Qf6 Qxh4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7 Bf2 Nc6 Nd5 Rb8 O-O e6 Nxc6 bxc6 Ne3 d5 Rb1 O-O c4 d4 Nc2 e5 Ne1 f5 Nd3 g4 Nc5 gxf3 Qxf3 fxe4 Qxh5 Qe8 Qg5 Rf5 Qd2 e3 Bxe3 Rxf1 Rxf1 dxe3 Qxd7 Qxd7 Nxd7 Rxb2 Nf6 Bxf6 Rxf6 Rxa2 Rf1 a5 Re1 e2 Kf2 a4 Rb1 a3 Rb8 Kf7 Ra8 Ra1 Kxe2 a2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 Nc6 Nf3 e6 a3 d5 Nc3 a6 cxd5 exd5 Bg5 Be7 e3 h6 Bh4 O-O Bd3 Be6 O-O Nd7 Bg3 Bd6 Rc1 Bxg3 hxg3 Ne7 Na4 c6 Qc2 a5 b4 b5 Nc5 Nb6 bxa5 Nc4 a6 Bc8 a4 Bxa6 Ra1 Qd6 axb5 Bxb5 Qc3 Qc7 Rfb1 Nd6 Bc2 Nb7 Nxb7 Qxb7 Qc5 Rfb8 Ne5 Rxa1 Rxa1 Nc8 g4 Nb6 Bf5 Rd8 Rb1 Na4 Qc2 Nb6 Bh7 Kh8 Bd3 Bxd3 Qxd3 f6 Ng6 Kg8 Qf5 Re8 Nf4 Qc7 Qg6 Re7 Nh5 Nd7 Rc1 Nf8 Qc2 Re6 Nf4 Rd6 Ra1 Qc8 Qc5 Rd8 Qb6 Rd6 Ra7 Qxg4 Qb8 Rd7 f3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 a6 Qd2 Nbd7 d5 Ne5 Rd1 c6 dxc6 bxc6 b3 Qa5 Na4 Qc7 Ne2 Nh5 Nec3 f5 exf5 gxf5 f4 Ng4 Be2 h6 Bh4 d5 Bxg4 fxg4 O-O d4 Ne2 c5 Nb2 a5 Rde1 Bf5 Nd3 a4 Ne5 axb3 axb3 Bf6 Bxf6 exf6 Nd3 Ra3 Nec1 Ng7 Re2 h5 Rfe1 Rf7 Qb2 Qa7 Nf2 h4 Ne4 Ra6 Nd3 Bxe4 Rxe4 Ra2 Qb1 h3 gxh3 gxh3 Nf2 Qa5 Qd3 Ra1 Kf1 f5 Rxa1 Qxa1 Re1 Qa8 Qxh3 Ra7 Qg2 Qxg2 Kxg2 Ra3 Re5 Rxb3 Rxc5 Rc3 Rd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Kh1 b6 Bg5 Bb7 Bxf6 Bxf6 Bc4 Nc6 Bd5 Rc8 Nd2 b5 a3 Qc7 Nf3 Rfd8 Qd2 b4 axb4 Nxb4 Bb3 Bc6 Rfd1 a5 Ne1 Ra8 Nd3 Qb6 f3 h6 Nxb4 axb4 Nd5 Bxd5 Qxd5 Ra7 Rxa7 Qxa7 Qc4 Rb8 g3 Kh7 Kg2 Qe3 Rxd6 Rc8 Qf1 Rd8 Rxd8 Bxd8 Bxf7 Qd2 Qf2 Qxf2 Kxf2 Bb6 Ke2 Bd4 c3 bxc3 b3 g5 Kd3 Kg7 Be8 Kf6 b4 Bg1 h4 gxh4 gxh4 Bf2 h5 Ke7 Ba4 Bg3 Kxc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Be7 cxd5 exd5 Bf4 Nf6 e3 Bf5 Qb3 Nc6 g4 Nxg4 Nxd5 O-O Bg2 Bh4 Bg3 Be6 Kf1 a5 Nh3 a4 Qc4 Na5 Qc5 b6 Bxh4 bxc5 Bxd8 Raxd8 Ne7 Kh8 d5 Bd7 Rc1 c4 Ng5 h6 Nf3 Rb8 h3 Nf6 Nd4 Rxb2 Bf3 Rxa2 Kg2 Re8 Nec6 Bxc6 Nxc6 Nxc6 dxc6 Rxe3 Rxc4 Rb3 Rd4 Rbb2 Rf1 Rd2 Rb4 Kh7 Rb7 Ne8 Kg3 g6 Re1 Nd6 Rxc7 Kg7 Rd7 Nf5 Kf4 Rxf2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7 O-O Nc6 Bf2 e6 Nce2 Ne5 b3 g4 f4 h4 Be3 h3 g3 Nc6 Qd3 O-O Rad1 f5 c4 Qa5 Nc3 Rae8 Rfe1 e5 Nxc6 Bxc6 b4 Qa3 b5 exf4 Bxf4 axb5 cxb5 Qc5 Be3 Qxc3 bxc6 Qxc6 Qxd6 Qxe4 Qd5 Qxd5 Rxd5 Bc3 Re2 Re4 Kf2 Rfe8 Rd3 Bf6 Red2 Rxe3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 g3 Bg4 b3 Nd7 Bb2 e6 Bg2 Ngf6 O-O c6 d3 Bd6 Nbd2 O-O h3 Bh5 e3 h6 Qe1 Qa5 a3 Bc7 Nh4 g5 Nhf3 e5 e4 Rfe8 Nh2 Qb6 Qc1 a5 Re1 Bd6 Ndf1 dxe4 dxe4 Bc5 Ne3 Rad8 Nhf1 g4 hxg4 Nxg4 f3 Nxe3 Nxe3 Be7 Kh1 Bg5 Re2 a4 b4 f5 exf5 e4 f4 Bxe2 fxg5 Ne5 g6 Bf3 Bc3 Qb5 Qf1 Qxf1 Rxf1 h5 Kg1 Kf8 Bh3 b5 Kf2 Kg7 g4 Kh6 Rg1 hxg4 Bxg4 Bxg4 Nxg4 Nxg4 Rxg4 Rd5 f6 Rd1 g7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 h6 d4 Re8 Nbd2 Bf8 Nf1 Bd7 Ng3 Na5 Bc2 c5 b3 Nc6 d5 Ne7 Be3 Ng6 Qd2 Nh7 a4 Nh4 Nxh4 Qxh4 Qe2 Qd8 b4 Qc7 Rec1 c4 Ra3 Rec8 Rca1 Qd8 f4 Nf6 fxe5 dxe5 Qf1 Ne8 Qf2 Nd6 Bb6 Qe8 R3a2 Be7 Bc5 Bf8 Nf5 Bxf5 exf5 f6 Bxd6 Bxd6 axb5 axb5 Be4 Rxa2 Qxa2 Qd7 Qa7 Rc7 Qb6 Rb7 Ra8 Kf7 Qa6 Qc7 Qc6 Qb6 Kf1 Rb8 Ra6 h5
Result: 1-0
Processing game...
FEN: 
Moves: d3 e5 Nf3 Nc6 c4 Nf6 a3 d6 Nc3 Be7 g3 O-O Bg2 Be6 O-O Qd7 Ng5 Bf5 e4 Bg4 f3 Bh5 Nh3 Nd4 Nf2 h6 Be3 c5 b4 b6 Rb1 Kh8 Rb2 a6 bxc5 bxc5 Bh3 Qc7 Bg4 Bg6 f4 exf4 gxf4 Qa5 Bd2 Qxa3 Ra2 Qb3 f5 Qxd1 Bxd1 Bh7 Nh3 Rfb8 Nf4 Bd8 Nfd5 Nc6 Bf4 Ne5 Ba4 Nxd5 Nxd5 a5 Bb5 Ra7 Kg2 g5 Bxe5 dxe5 f6 Bg6 h4 gxh4 Kh3 Kg8 Kxh4 Kh7 Kg4 Bc7 Nxc7 Rxc7 Rxa5 Rd8 Rf3 Kh8 Kh4 Kg8 Ra3 Kh8 Ra6 Kh7 Ra3 Kh8 Ra6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Nb3 Be7 Qf3 Qc7 O-O-O Nbd7 g4 b5 Bxf6 Nxf6 g5 Nd7 a3 Rb8 h4 b4 axb4 Rxb4 Na2 Rb6 Qc3 Qxc3 Nxc3 h6 Be2 Bb7 Na5 Ba8 Nc4 Rc6 Kb1 Nb6 b3 Nxc4 Bxc4 Rc5 Kb2 a5 Bd3 Bc6 Na4 Bxa4 bxa4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Qc7 Qg4 f5 Qg3 cxd4 cxd4 Ne7 Bd2 O-O Bd3 b6 Ne2 Ba6 Nf4 Qd7 h4 Bxd3 Qxd3 Nbc6 Rh3 Rac8 Rg3 Rf7 h5 Nd8 c3 Rf8 Kf1 Rc4 Kg1 Nf7 a4 Rfc8 Qb1 Nc6 Qd1 Ne7 h6 g6 Qh5 Rxa4 Rxa4 Qxa4 Nxe6 Qc2 Qh4 f4 Bxf4 Nf5 Rxg6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bc5 a4 Rb8 axb5 axb5 Nxe5 Nxe5 d4 Bxd4 Qxd4 d6 f4 Nc6 Qc3 Ne7 Qd3 O-O Nc3 c5 Nxb5 Nxe4 Qxe4 Rxb5 Ra7 c4 Bxc4 d5 Qxe7 dxc4 Qxd8 Rxd8 Ra8 Rf8 Ra4 Be6 f5 Bd5 Re1 h5 h3 Rc8 Kh2 h4 f6 gxf6 b3 f5 bxc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 O-O Be7 Bb3 O-O f4 b5 e5 dxe5 fxe5 Nfd7 Be3 Nxe5 Qh5 Nbc6 Nxc6 Nxc6 Rf3 b4 Rh3 h6 Rd1 Qa5 Nd5 exd5 Rg3 d4 Bd5 Bg5 Bxg5 Qxd5 Bf6 Qxh5 Rxg7 Kh8 Rg6 Kh7 Rg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 c5 e3 d6 dxc5 bxc5 O-O-O Ne4 Qd3 Nxf2 Bxd8 Nxd3 Bxd3 Rxd8 Be4 d5 cxd5 Bb7 Ne2 Nd7 dxe6 Bxe4 e7 Re8 Rxd7 f6 Ng3 Bg6 h4 h6 h5 Bh7 Rh4 Kf7 Rc4 Rxe7 Rxe7 Kxe7 Rxc5 Kd6 b4 Re8 Kd2 Re5 Rxe5 Kxe5 a4 Bg8 b5 Bb3 a5 Bc4 b6 axb6 axb6 Bd5 e4 Bb7 Ke3 Bc6 Nf5 Bxe4 g4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Be2 Nbd7 d3 b6 O-O Bb7 Nbd2 g6 d4 cxd4 cxd4 Nxe4 Nxe4 Bxe4 Ng5 d5 Bb5 Bg7 f3 Bf5 g4 h6 gxf5 hxg5 fxg6 a6 gxf7 Kxf7 Ba4 Rh5 Be3 Nf6 Qd2 Qd6 Rf2 Rah8 Rg2 Rh3 Rf1 R8h4 Bc2 Nh5 Bf5 Nf4 Bxh3 Nxh3 Kh1 Qf6 Rg3 Qf5 Bxg5 Nxg5 Rxg5 Qh3 Rg2 Bf6 Qd3 Rxd4 Qg6 Ke6 Qe8 Rc4 Qd8 Qf5 Re1 Be5 Qb8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 c5 d5 Ba6 Qc2 exd5 cxd5 g6 Nc3 Bg7 g3 O-O Bg2 d6 O-O Re8 Re1 Nbd7 h3 b5 e4 Qc8 Bf4 b4 Na4 b3 Qxb3 Nxe4 Qc2 Ndf6 g4 Qd7 g5 Nh5 Bh2 f5 Nc3 Rab8 Rab1 Bxc3 bxc3 Rxb1 Rxb1 Bc4 Nd2 Nxd2 Qxd2 f4 Re1 Re5 Re4 Rxe4 Bxe4 Qxh3 Bg2 Qg4 Qe1 Ng7 f3 Qxg5 Qb1 Nf5 Qb8 Kg7 Qxa7 Kh6 Qf7 Bf1 Kxf1 Ne3 Ke1 Qh4 Ke2 Qxh2 Kd3 Nf5
Result: 0-1
Processing game...
FEN: 
Moves: c4 e5 Nc3 Nc6 e3 Nf6 a3 g6 b4 Bg7 Bb2 O-O d3 Re8 Qc2 d5 cxd5 Nxd5 Nxd5 Qxd5 Ne2 f5 Nc3 Qf7 Na4 Be6 Nc5 Bd5 b5 Nd4 exd4 exd4 Be2 Bxg2 O-O-O Bxh1 Rxh1 Qd5 Re1 Re5 Qb3 Qxb3 Nxb3 Rae8 Kd1 Rxb5 Bf3 Rxe1 Kxe1 c6 Bd1 a5 Bxd4 a4 Bxg7 Kxg7 Nd2 Re5 Be2 b5 Kd1 Rd5 Kc2 g5 Bf3 Rd6 h3 Kg6 Nb1 h5 Nc3 g4 Bg2 Kf6 hxg4 hxg4 d4 Kg5 Kd3 Rh6 Ke2 f4 Be4 Rh3 Kd2 Rh2 Ke1 g3 fxg3 fxg3 Kf1 Rf2 Kg1 b4 axb4 a3 d5 Kf4 Bg6 cxd5 Nxd5 Kg5
Result: 0-1
Processing game...
FEN: 
Moves: c4 e6 Nc3 d5 d4 Bb4 e3 c5 a3 Bxc3 bxc3 Nf6 cxd5 exd5 f3 c4 Ne2 Nc6 g4 h6 Bg2 Na5 O-O Nb3 Ra2 O-O Ng3 Bd7 Qe1 Re8 e4 dxe4 fxe4 Nxg4 Bf4 Qh4 h3 Nf6 e5 Rad8 Qf2 Nh5 Bxh6 Re7 Nf5 Qxf2 Rfxf2 Re6 Be3 Bc6 Bf1 f6 Bxc4 Bd5 Be2 fxe5 Bxh5 exd4 Bg5 Rd7 Rae2 Be4 Nxd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 Qa5 Rb1 b6 Bb5 Bd7 Be2 Bc6 Bd3 Nd7 Ne2 Rd8 f3 O-O h4 h5 Bg5 Rfe8 Rc1 Bb7 d5 Ne5 Bb1 Nc4 Qf4 Be5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 Nbd7 g4 h6 O-O-O b5 h4 Nb6 Kb1 b4 Bxb6 Qxb6 Nd5 Bxd5 exd5 a5 Qd3 O-O Nd2 a4 Ne4 Rfc8 Nxf6 Bxf6 Qe4 Qc5 Bd3 Kf8 g5 hxg5 hxg5 Bxg5 Qf5 Bh6 Rh4 Ke7 Rc4 Qxc4 Bxc4 Rxc4 Qd3 Rac8 Re1 Bf4 Re4 Rxe4 fxe4 g5 a3 bxa3 Qa6 Rd8 Qb6 g4 c4 g3 c5 g2 cxd6 Rxd6 Qc7 Kf6 Qxd6 Kg7
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nc3 e6 Nf3 a6 g3 b5 Bg2 Bb7 d4 cxd4 Nxd4 b4 Na4 Nf6 Bg5 Qa5 Bxf6 gxf6 c3 Nc6 O-O Bg7 cxb4 Qxb4 Ne2 Bh6 f4 O-O Rc1 a5 a3 Qe7 Nc5 Ba6 Rf2 Ra7 Nxa6 Rxa6 Nc3 Rb8 Rd2 Ra7 a4 Rb4 Na2 Rb8 Nc3 Rb4 Na2 Rb8 Nc3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 Nc3 d5 cxd5 Nxd5 e4 Nb4 Bb5 N8c6 d4 cxd4 a3 dxc3 Qxd8 Kxd8 axb4 cxb2 Bxb2 e6 O-O Bd7 Bxc6 Bxc6 Ne5 Ke8 Nxc6 bxc6 Ra4 f6 Rfa1 Kf7 Rxa7 Rxa7 Rxa7 Be7 Rc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 h3 Bh5 O-O Nc6 Be3 cxd4 Nxd4 Bxe2 Qxe2 Nxd4 Bxd4 Be7 Rd1 Qc6 c4 O-O Nc3 a6 b3 Rfd8 a4 Rd7 Rd3 Rad8 Rad1 h6 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 c4 Nc6 O-O g6 d4 cxd4 Nxd4 Bg7 Nc2 Rc8 Ne3 Ne5 Qe2 Qc6 Nd2 Nf6 f4 Ned7 Kh1 O-O Rb1 a5 Nd5 Nxd5 exd5 Qa4 a3 Qc2 b3 Nc5 Qd1 Qxd1 Rxd1 a4 bxa4 Ra8 a5 Rxa5 Nb3 Nxb3 Rxb3 Rc8 Rxb7 Bf6 g4 h6 g5 hxg5 fxg5 Be5 Rxe7 Rxc4 Be3 Rxa3 Rxe5 dxe5 d6 Rxe3 d7 Re1 Rxe1 Rd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 Nc3 Ne4 Bd2 Bf6 Qc2 Nxd2 Qxd2 d6 d5 O-O Nd4 Qe7 O-O Bxd4 Qxd4 e5 Qd2 f5 f4 exf4 gxf4 Nd7 e4 fxe4 Nxe4 h6 Rae1 Qh4 Qe3 Nf6 Qg3 Qxg3 Nxf6 Rxf6 hxg3 Bc8 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bg5 Ne4 Bh4 Nxc3 bxc3 dxc4 e3 b5 a4 c6 Be2 a6 Nd2 O-O Bf3 Ra7 O-O Bf5 Re1 Bd3 Nb3 Bf5 Nd2 Bd3 g4 Rc7 Nb3 cxb3 Qxd3 c5 Bg3 e5 axb5 c4 Qe4 axb5 Ra8 Qe7 dxe5 Nd7 e6 Nc5 exf7 Qxf7 Rxf8 Bxf8 Qa8 Ra7 Qc6 Ra6 Qd5 b2
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 dxc4 e3 e6 Bxc4 c5 O-O a6 Bb3 Nc6 Nc3 Be7 Qe2 O-O Rd1 cxd4 Nxd4 Nxd4 exd4 Nd5 Qf3 Nxc3 bxc3 Qc7 c4 Bd6 c5 Be7 Bf4 Qd7 Rac1 Qc6 d5 exd5 Bxd5 Qg6 h3 Ra7 Qe3 Bf6 c6 bxc6 Be4 Bf5 Bxf5 Qxf5 g4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nc6 d4 cxd4 Nxd4 e6 Nc3 Nf6 a3 Be7 e4 O-O Be2 d5 exd5 exd5 O-O Nxd4 Qxd4 Be6 cxd5 Nxd5 Nxd5 Qxd5 Be3 Bf6 Qxd5 Bxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Nbd7 O-O O-O Re1 c5 e4 dxc4 Nxc4 Bb7 e5 Nd5 Bd2 Rc8 a4 a6 Qe2 Ba8 Rac1 cxd4 Nxd4 Nc5 Red1 Qd7 h4 h6 Qg4 Kh7 Qh5 f5 Bg5 Ne4 Nxb6 Nxb6 Nxf5 Rxf5 Rxd7 Rxc1 Bxc1 Rxh5 Rxe7 Nc5 Bxa8 Nxa8 Be3 Rxe5 Bd4 Rd5 Rxg7 Kh8 Rd7 Rxd4 Rxd4 Nb6 Rd6 Nd5 Rc6 Nxb3 Rxe6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 g3 e5 Nde2 Be7 Bg2 b5 O-O Bb7 h3 Nbd7 g4 b4 Nd5 Nxd5 exd5 O-O Ng3 Re8 a3 a5 Be3 Ba6 Re1 Qb8 axb4 axb4 b3 Rc8 Qd2 Nc5 Ra2 Bb7 Rea1 Rxa2 Rxa2 Ba8 Ne4 Nd7 Ra4 Qc7 Ra7 Qd8 Ng3 Bf8 Qxb4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be7 f3 Be6 Qd2 O-O O-O-O b5 g4 b4 Na4 d5 Bc5 d4 Qxb4 Nc6 Bxe7 Qe8 Qa3 Nxe7 Nbc5 Ng6 c3 Nd7 cxd4 exd4 Kb1 Nxc5 Nxc5 Qc6 Nxe6 fxe6 Rxd4 Ne5 Be2 Rac8 Rd2 Rf7 Qxa6 Nxf3 Rd8
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 O-O O-O Ne5 Be2 Qg6 Nd2 d6 Bh5 Qf6 f4 N5c6 Kh1 Nxd4 cxd4 Bb6 Nc4 d5 e5 Qe6 Nxb6 axb6 Bf3 f5 Qb3 Bd7 Rfc1 Bc6 Bd2 Ng6 Qe3 Rfe8 b3 Qd7 a4 Nf8 Bb4 Ne6 g4 Qf7 Rg1 Kh8 Raf1 Rad8 Rg3 Rd7 Kg1 g6 gxf5 Qxf5 Bg4 Qe4 Qxe4 dxe4 Bxe6 Rxe6 f5 gxf5 Rxf5 h6 Rf6 Bd5 Bf8 Rxf6 exf6 Be6 Kf2 Kh7 Be7 h5 b4 Rd5 Ke3 Rf5 Rg7 Kh8 Kxe4 c6 Bd8 Rf2 Ke5 Re2 Kd6 Bb3 Rxb7 Rxh2 a5 bxa5 bxa5 Bc4 Rb4 Bf1 d5 cxd5 Rf4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nbd7 f4 Nc5 f5 Be7 O-O O-O fxe6 fxe6 Nf5 Nxb3 Nxe7 Qxe7 axb3 b5 Bf4 e5 Bg5 Bb7 Qd3 Qe6 Bxf6 Rxf6 Rxf6 Qxf6 Nd5 Qh4 Rd1 Rf8 Qe3 h6 h3 Bxd5 Rxd5 Rf4 Rxd6 Rxe4 Rd8 Kh7 Qd3 Qe1 Kh2 Qe3 Qxe3 Rxe3 Rd6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 e3 Bf5 Nbd2 e6 Be2 c6 O-O Bd6 c4 Nbd7 Re1 Ne4 Nxe4 dxe4 Nd2 Nf6 f3 Qb8 Nf1 h5 fxe4 Nxe4 d5 h4 Bf3 exd5 cxd5 O-O Bg4 Qc8 Bxf5 Qxf5 Qf3 Qxd5 Rd1 Qe6 Rd4 f5 Nd2 Rae8 Qh3 Ng5 Qxh4 Qxe3 Kf1 Qe2 Kg1 Bc5 Nb3 Qd1 Kf2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5 Nb5 d6 N1c3 a6 Na3 b5 Nd5 Nce7 Be3 Nxd5 Qxd5 Be6 Qd2 Nf6 Bd3 d5 exd5 Qxd5 O-O e4 Be2 Qxd2 Bxd2 Rc8 Rac1 Bxa3 bxa3 Bc4 Bxc4 Rxc4 Bb4 Nd5 c3 Kd7 Rfd1 Ke6 Rd4 Rhc8 Rxc4 Rxc4 g3 g5 Kf1 f5 Ke2 h5 Kd2 Rc8 Re1 Nb6 a4 bxa4 Ke3 Rd8 Bc5 Nc4 Ke2 Rd2 Kf1 Rxa2 Rd1 e3 fxe3 Nd2 Ke1 Nf3 Kf1 Nxh2 Ke1 Nf3 Kf1 Nd2 Ke1 Ne4 Bb4 h4 gxh4 gxh4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 a6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Nb3 d6 g3 Be7 Bg2 Nbd7 O-O O-O a4 b6 f4 Bb7 g4 Rc8 g5 Ne8 Qg4 exf4 Bxf4 Ne5 Qg3 Ng6 Be3 d5 e5 Bb4 Bd4 Nc7 Bxb6 Qe7 Na2 Na8 Nxb4 Nxb6 c3 Nc4 Rf2 d4 Bxb7 Qxb7 Nxd4 Ncxe5 h4 Rc5 a5 Qc8 Nf5 Rc4 Qxe5 Rg4 Kf1 Nxe5 Ne7 Kh8 Nxc8 Rxh4 Re1 Rh1 Ke2 Rxe1 Kxe1 Rxc8 Nxa6 Rc5 Rf5 Rxa5 Kd1 g6 Rf6 Ng4 Rc6 Rxg5 Rc5 Rxc5 Nxc5 Kg8 b4 Kf8 Ke2 Ke7 b5 Ne5 Ke3 h5 Kd4 Nd7 Nd3 h4 Ke3 f5 Kf4 Kf6 c4 Ne5 Nb2
Result: 0-1
Processing game...
FEN: 
Moves: d4 e6 g3 Nf6 Bg2 d5 Nf3 Be7 O-O O-O c4 c6 Qc2 b6 b3 a5 Nc3 Bb7 Bf4 Na6 Rad1 Ra7 Ne5 Ne8 e4 f6 Nf3 dxe4 Nxe4 Nd6 Rfe1 Re8 Nc3 Nb4 Qb1 Bf8 c5 Nf7 cxb6 Qxb6 d5 Nxd5 Na4 Qb4 Bd2 Qe7 Nd4 Qd7 Qc1 Ne5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 d5 Bg2 dxc4 Ne5 Bb4 Kf1 Nd5 Nxc4 Bd6 e4 Ne7 Bb2 Nbc6 Nd2 e5 d5 Na5 Kg1 Nxc4 Nxc4 Bxc4 bxc4 O-O Bh3 f5 Kg2 Qe8 Rf1 f4 f3 Kh8 a4 a5 Ba3 Qg6 Qc1 Rab8 Rb1 h5 Bxd6 cxd6 Be6 Qg5 Kh1 Qh6 g4 hxg4 Bxg4 Ng8 Rg1 Nf6 Bf5 g5 Rg2 Nh5 Kg1 Qf6 Bg4 Ng7 Qe1 Qe7 Qf2 Qd8 h4 gxh4 Rh2 Kg8 Rxh4 Qg5 Rh2 Kf7 Rxb6 Qd8 Rb5 Qc7 Rh6 Rh8 Rxh8 Rxh8 c5 Qa7 Rb6 Rb8 Qb2 Rxb6 Qxb6 Qe7 cxd6 Qh4 d7 Qe1 Kg2 Qe2 Qf2 Qxf2 Kxf2 Ke7 Ke2
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 Bg5 d5 e3 c5 Nbd2 Nc6 c3 Bd7 Bd3 Be7 O-O b6 Ne5 O-O f4 h6 Bh4 Ne8 Bf2 Nxe5 dxe5 f5 exf6 Nxf6 Nf3 Qe8 a4 Ng4 Qc2 Bf6 Bh7 Kh8 Bg6 Qe7 e4 Nxf2 Rxf2 dxe4 Bxe4 Rac8 Re1 Bc6 Bxc6 Rxc6 Qe4 Rd6 Qc4 Qd7 Rfe2 Re8 h3 Re7 Kh2 Kh7 Ne5 Bxe5 Rxe5 g6 R1e2 Kg7 R5e4 Kf6 a5 Kg7 axb6 axb6 b4 Rd5 Re5 Rxe5 Rxe5 cxb4 cxb4 Kf6 h4 Qd6 h5 gxh5 Rxh5 Kg6 Re5 Kf6 Qc3 Kg6 b5 Rf7 Qe3 Rf6 g3 Kf7 Qe4 Qd2 Kh3 Qf2 Qh7 Kf8 Qd3 Kg7 Re2 Qf1 Kh2 Rf5 Qd7 Kf8 Qd6 Kg7 Qe7 Kh8 Qe8 Kg7 Rxe6 Qf2 Kh3 Qf1 Kg4 Rf7 Qc6 Qd1 Kh4 Qd8 Kh3
Result: 1-0
Processing game...
FEN: 
Moves: c4 g6 e4 Bg7 d4 Nc6 Nf3 d6 d5 Ne5 Nxe5 Bxe5 Bd3 Bg7 Nc3 Nf6 O-O O-O h3 e6 Bg5 c6 Qd2 cxd5 cxd5 e5 a4 a6 a5 Bd7 Na4 Bxa4 Rxa4 Rc8 Rc1 Qd7 Rac4 Rxc4 Rxc4 Rc8 Rxc8 Qxc8 Qc2 Qxc2 Bxc2 h6 Be3 Nd7 b4 Kh7 Kf1 g5 Ke2 Kg6 Ba4 Nf6 Kd3 Bf8 Bb6 Be7 Bc6 g4 Bxb7 gxh3 gxh3 Nh5 b5 Nf4 Kc2 axb5 a6 Nxh3 a7 h5 a8=Q Nf4 Qg8 Kf6 Bc6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 c3 d5 exd5 Qxd5 d4 Nc6 Nf3 Bg4 Be2 Nf6 O-O e6 h3 Bh5 Be3 cxd4 Nxd4 Bxe2 Qxe2 Nxd4 Bxd4 Be7 Nd2 O-O Rfd1 Qc6 Nf3 Rfd8 c4 Qe4 Kf1 Qxe2 Kxe2 Ne8 Rd2 f6 Rad1 Kf7 Be3 Rxd2 Rxd2 e5 Ne1 Nd6 Nd3 b6 Rc2 Nf5 g4 Nxe3 Kxe3 h5 f4 Bc5 Kf3 hxg4 hxg4 exf4 Nxf4 Re8 Nd5 Bd6 Rd2 Be5 b3 Rh8 Nc7 Ke7 Nb5 Rh3 Ke4 Ke6 Nxa7 Rg3 Nc6 Rxg4 Kf3 Rg3 Kf2 Rh3 Nxe5 Kxe5 Rd5 Ke4 Kg2 Rc3 Rb5 Rc2 Kh3 Rxa2 Rxb6 f5 Rd6 g5 b4 Ra4 Rb6 Kf3 c5 Ra7 Kh2 g4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Rb8 Qd2 Na5 Qf2 Nc4 Bc1 e5 Nde2 exf4 Nxf4 Be6 b3 Ne5 Bb2 Rbc8 Rac1 Qc5 Qg3 g6 Nce2 Nxf3 gxf3 b5 axb5 axb5 Bd4 Qc6 Qg2 b4 Ng3 Qb5 Nxe6 fxe6 f4 e5 Bb2 Rc5 f5 g5 Rce1 Qc6 Re2 Kf7 Bc1 Rg8 Be3 Rc3 Bd2 Rxc2 Bxb4 Rxe2 Qxe2 h5 Nxh5 Nxe4 Qf3 g4 Qg2 Rh8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5 f3 f4 Bf2 g5 Rc1 Ng6 c5 Nxc5 b4 Na6 Nd3 h5 Nb5 Bd7 a4 Bh6 Rc3 b6 Be1 Rf7 Nf2 Nh4 Nxd6 cxd6 Bxa6 Qe8 Qe2 g4 fxg4 Rg7 h3 Qg6 Bb5 Bxb5 axb5 Rf8 Nd1 hxg4 Bxh4 f3 Qc2 gxh3 g3 Rf4 Ne3 Rxe4 Nf5 Qxf5 Rcxf3 Qg4 Kh1 Rf4 R3f2 Rxf2 Rxf2 e4 Rf6 e3 Re6 Qf3 Kg1 Rf7 Qg6 Bg7 Re8 Rf8 Rxf8 Kxf8 Be7 Kg8 Qe6 Qf7 Qc8 Bf8 Qg4 Bg7 Qc8 Kh7 Qxh3 Bh6 Bg5 Qf2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g5 g3 Bg7 Bb2 O-O Nd2 f6 Qh5 Nb4 h4 g4 Kd1 c5 a3 Nc6 Bd3 f5 Bxf5 Bxe5 Re1 d6 Be4 Bb7 Qxg4 Qg7 Bd5 Kh8 Bxe5 dxe5 Qxg7 Kxg7 Ne4 Rad8 Nxc5 Bc8 Ra2
Result: 1-0
Processing game...
FEN: 
Moves: e4 Nf6 e5 Nd5 d4 d6 Nf3 dxe5 Nxe5 g6 Bc4 c6 O-O Bg7 Re1 O-O Bb3 Nd7 Nf3 N7f6 c4 Nc7 h3 c5 Nc3 cxd4 Nxd4 Ne6 Be3 Nxd4 Qxd4 Be6 Qh4 Qa5 c5 Bxb3 axb3 Qc7 b4 a6 Bf4 Qc6 Rxe7 Nd5 Nxd5 Qxd5 Qg4 Rae8 Qd7 Qxd7 Rxd7 Bxb2 Rad1 Re4 Bd6 Rc8 Rxb7 Bc3 Rb1 Rc4 Rc1 Rxb4 c6 Rd4 c7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Bg5 a6 Qd2 c5 d5 b5 cxb5 Nbd7 a4 Qa5 Nge2 Nb6 Nc1 axb5 Bxb5 Ba6 N1a2 Bxb5 axb5 Nh5 Rb1 Bd4 Bh6 Rfe8 b3 e6 dxe6 Rxe6 Be3 Bxe3 Qxe3 d5 b4 Qa3 bxc5 Nc4 Qd4 Nf4 O-O Qxa2 Rf2 Qa3 Nxd5 Qd3 Qxd3 Nxd3 Rc2 Na3 Ra2 Nxc5 Rba1 f5 Nc7 Re5 Nxa8 Nxb5 exf5 gxf5 Nb6 Nc3 Rc2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Be3 Bb4 Na4 Bd6 g3 Be7 c4 d6 f3 Bd7 Rc1 O-O Nxc6 Bxc6 Nb6 Rad8 b4 Qb8 b5 Be8 a4 Nd7 Nxd7 Rxd7 Qb3 Qc7 Rfd1 Rd8 Bd4 Ra8 Qc3 axb5 cxb5 Qxc3 Rxc3 Rxa4 Rc7 Bg5 Bf2 b6 Rb7 Ra2 Bf1 Rb2 f4 Bf6 Rxb6 Bc3 Rb7 g5 f5 Bb4 f6 h5 Rc1 Kh7 Rb8 d5 exd5 Bd6 Rd8 Be5 Rcc8 Bxb5 Bxb5 Rxd8 Bd3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 c5 Nf3 d5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7 Ba2 b6 Re1 e5 e4 Bg4 dxc5 bxc5 h3 Rad8 Qe2 Bxf3 Qxf3 Rd6 Bg5 h6 Bh4 Rfd8 Rab1 Ne7 Bc4 Nc8 Bg3 Nb6 Bb5 Re6 a4 c4 Qe2 Rd3 a5 Nc8 Rb4 Rxc3 Rxc4 Rxc4 Qxc4 Qb8 Qc5 Nd6 Bd3 Nd7 Qa3 Nf8 Rb1 Qc7 Rc1 Qd8 a6 Ng6 Qc5 Re7 f3 Ne8 Bf1 Rc7 Qe3 Rd7 Kh2 Re7 Rc6 Kh7 Qc1 Nc7 Qc3 Qd7 Rc5 Qd6 Bf2 Ne6 Rd5 Qb8 Rb5 Qd6 Rb7 Nd4 Qb4 Qf6 Qc5 Nc6 Be3 Re6 Bc4 Re7 Bd5 Nd4 Rxa7 Rxa7 Qxa7 Ne7 Bc4 h5 Qc5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 Nd4 Nxd4 cxd4 O-O g6 c3 dxc3 Nxc3 Bg7 d4 e6 d5 Ne7 d6 Nc6 Be3 O-O Qd2 f5 Rad1 Ne5 f4 Ng4 Bd4 fxe4 Nxe4 Nf6 Nxf6 Bxf6 Kh1 a6 Be2 Bxd4 Qxd4 b5 f5 gxf5 Rd3 Rf6 Bh5 Bb7 Rg3 Kf8 Qf4 Rc8 Qg5 Rg6 Qxd8 Rxd8 Bxg6 hxg6 Rxg6 Rc8 Kg1 Rc2 h4 Kf7 h5 Be4 Rg3 e5 h6 f4 Rg7 Kf6 Rf2 Rc1 Kh2 Bf5 g3 Be4 gxf4 Rh1 Kg3 Rg1 Kh3 Rh1 Rh2 Bf5 Kg2 Be4 Kg3 exf4 Kxf4 Rxh2 Kxe4 Rxh6 Rxd7 Ke6 Ra7 Rh2 Rxa6 Rxb2 Kd4 Rc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Nf3 e5 O-O Nc6 d5 Ne7 b4 a5 Ba3 axb4 Bxb4 Nd7 a4 f5 a5 Nf6 c5 Nxe4 Nxe4 fxe4 Ng5 e3 cxd6 cxd6 fxe3 Bh6 Rxf8 Qxf8 Ne4 Bxe3 Kh1 Nf5 Ra3 Bd4 Qd2 Qd8 Bg4 Ne7 Bxc8 Qxc8 Nxd6 Qd7 Rf3 Nxd5 Qa2 Qc6 Rf1 Rf8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 Qe2 c5 g3 Nc6 Nf3 g6 Bg2 Bg7 O-O Nge7 d3 O-O c3 b6 Re1 d6 Nbd2 Bb7 Nf1 b5 h4 h6 N1h2 Kh7 Bd2 Qd7 d4 cxd4 cxd4 a6 Rad1 Rac8 Bf4 Rfe8 Qd2 Qc7 Ng4 Ng8 e5 dxe5 dxe5 Red8 Qe3 Qe7 Rd6 Nb4 Ng5 Kh8 Bxb7 Qxb7 Qb6 Qxb6 Nxf7 Kh7 Rxb6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 Nf6 O-O
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 c5 Bxf6 gxf6 d5 Qb6 Qc1 Bg7 c3 d6 e3 Na6 Na3 Bd7 Ne2 O-O-O Qd2 e5 Ng3 f5 Nh5 Rhg8 Nxg7 Rxg7 g3 f4 Nc4 Qc7 exf4 Bg4 fxe5 Bf3 Bh3 Kb8 exd6 Rxd6 Nxd6 Qxd6 O-O Nc7 Rfe1 b5 Re3 Bxd5 Rae1 f6 b4 h5 bxc5 Qxc5 a4 bxa4 c4 Qxc4 Rb1 Ka8 Bg2 Rd7 Rc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nbd7 O-O e5 e4 c6 b3 Re8 h3 a6 d5 cxd5 cxd5 Nc5 Ne1 b5 b4 Ncd7 a4 bxa4 Rxa4 Nb6 Ra2 Bd7 Nc2 Qc8 Qd3 Bxh3 Ne3 Bd7 Bd2 Ng4 Nxg4 Bxg4 Rc1 Qb7 Rca1 Bc8 b5 axb5 Rxa8
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 g3 d5 Nd2 Nf6 Ndf3 e6 Nh3 Be7 Bg2 O-O O-O Ne4 Nf4 g5 Nd3 Qe8 Nfe5 Nd7 f3 Nd6 b3 Nf7 e4 Nfxe5 dxe5 fxe4 fxe4 Rxf1 Qxf1 c6 Bb2 Nc5 Nf2 Qh5 c4 dxe4 Bd4 Bd7 b4 Na6 Nxe4 Rf8 Nf6 Bxf6 exf6 b6 Rd1 Nxb4 Be5 Rf7 Rd6 c5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 c5 d5 b5 Nf3 g6 cxb5 a6 b6 d6 Nc3 Nbd7 a4 Qxb6 a5 Qa7 e4 Bg7 Be2 O-O O-O Rb8 Nd2 Ne8 Nc4 Nc7 Na4 Nb5 Nab6 Qb7 Rb1 Nf6 Bf3 Nd4 Bf4 Nxf3 Qxf3 Bg4 Qd3 Bd7 b4 cxb4 Rxb4 Bb5 Rfb1 Rfe8 h3 Nd7 Qe3 Red8 Bg5 Re8 Nxd7 Qxd7 Nb6 Qb7 Bh6 Bh8 Qg3 Rbd8 Bg5 Bg7 Rc1 h6 Bf4 Qb8 Bd2 e6 Bc3 e5 Bd2 Re7 Qh4 g5 Qg4 Rc7 Rbb1 Re8 h4 f6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 Nd2 Qe8 O-O Nh7 a3 Bd7 b3 Kh8 Rb1 h5 f3 Bh6 b4 axb4 axb4 g5 Bf2 Rg8 c5 Qe7 c6 bxc6 dxc6 Be6 g4 Rgb8 b5 Nc5 Qc2 Nf6 h3 Bf8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 c5 e3 Nc6 d5 Nb4 Nc3 Ng4 a3 Qa5 Bd2 Be7 h3 Nf6 axb4 Qxb4 Nb5 Qxb2 Bc3 Qxb5 cxb5 b6 d6 Bd8 Bd3 h6 e4 Nh5 O-O Bb7 Re1 Bf6 e5 Bd8 Be4 Bxe4 Rxe4 O-O Rea4 f6 g4 fxe5 gxh5 Bf6 Nxe5 Rac8 Rg4 c4 Rxa7 Rc5 Rxd7 Rd5 Qf3 Bxe5 Rdxg7 Bxg7 Rxg7 Kh8 Qxf8
Result: 1-0
Processing game...
FEN: 
Moves: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 O-O O-O h3 Nc6 Qe2 d5 c3 Na5 Bc2 Bd7 Rd1 Qc8 Nbd2 Nbc4 Nf1 b5 b3 Nb6 Bf4 Nb7 Bd3 c6 Rac1 Nd8 Qe3 Ne6 Bh6 a5 Ng3 f5 exf6 exf6 Bxg7 Kxg7 Re1 a4 b4 Qc7 Rcd1 Rae8 Qc1 Nf4 Bf1 Rxe1 Rxe1 Re8 Rxe8 Bxe8 Ne1 Nc8 Ne2 Nxe2 Bxe2 Qe7 Bf3 Nd6 Nd3 Bd7 Nc5 Bf5 g4 Bc8 Qe3 Qxe3 fxe3 f5 Kf2 Ne4 Bxe4 fxe4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 d6 d4 c6 Nc3 Nd7 f4 e6 Nf3 d5 e5 Bb4 Bd2 Ne7 a3 Bxc3 Bxc3 c5 Qd2 b6 b4 Bb7 Bd3 O-O O-O c4 Be2 b5 Qe1 Nb6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 d6 e4 O-O c3 Qe8 Be2 e5 dxe5 dxe5 O-O Nbd7 Qc2 Nc5 Rfe1 Ne6 Bh4 Nh5 Nc4 Nef4 Bf1 h6 a4 Kh8 a5 Rb8 b4 g5 Bg3 f5 Ne3 fxe4 Qxe4 Bd7 Rad1 Bc6 Qc4 a6 Ng4 e4 Nd4 Bd5 Qxc7 Rc8 Qb6 Rxc3 b5 axb5 Nxb5 Rb3 Qc5 Qf7 Nd6 Qe6 Rxd5 Qxd5 Qxd5 Nxd5 Bc4 Rd3 Bxd3 exd3 Rd1 Nhf4 Bxf4 Nxf4 g3 h5 gxf4 hxg4 fxg5 Bc3 Rxd3 Bxa5 Kg2 Bb6 Ne4 Re8 Nf6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 c5 Ngf3 Nc6 exd5 exd5 Bb5 Bd6 dxc5 Bxc5 O-O Ne7 Nb3 Bd6 Re1 O-O Bg5 Re8 c3 Bf5 Nfd4 Bd7 Bf1 Rc8 Bh4 Ne5 Bg3 f6 Nd2 Bb8 Qb3 Bc6 Rad1 Qa5 Qc2 Bd7 Qb1 a6 N2f3 Bg4 Be2 Bh5 Nxe5 Bxe2 Rxe2 fxe5 Bxe5 Nc6 Bxb8 Nxd4 Rxe8 Rxe8 Ba7 Ne2 Kh1 b6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Bd3 g6 Bc2 Bg7 h3 O-O d4 cxd4 cxd4 Nc6 Nc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e5 dxe5 Ng4 e3 Nxe5 Nc3 Nbc6 a3 Be7 f4 Ng6 g3 d6 Bg2 Be6 Nd5 Qd7 b4 Bg4 Qd3 O-O Nf3 a5 b5 Nd8 Bb2 Ne6 O-O Nc5 Qd4 f6 Rad1 Nb3 Qc3 Nc5 Ba1 Bd8 Qc2 Re8 Rfe1 Ne4 Rd4 Nc5 e4 c6 Ne3 Bxf3 Bxf3 Bb6 bxc6 bxc6 Rdd1 Qc7 Kh1 Rad8 Nf5 Ne7 Nd4 a4 Bg4 Ng6 h4 Ba5 Re2 Rb8 h5 Nf8 Re3 d5 cxd5 cxd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 Ne4 Bf4 c5 d5 Qb6 Bc1 e6 f3 Nf6 c4 d6 e4 Be7 Nc3 e5 Bd3 O-O Nge2 Nbd7 Qc2 a6 Be3 Qc7 g4 b5 b3 bxc4 bxc4 Rb8 Ng3 Rd8 Nf5 Nf8 O-O Ng6 Ne2 Bd7 Nxe7 Nxe7 h4 Rb4 Bd2 Rb7 Rab1 Rdb8 Rxb7 Rxb7 Rb1 Rxb1 Qxb1 Kf8 Kf2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 e5 Bf5 Nf3 e6 a3 Ne7 Nbd2 Nd7 Nh4 Bg6 Nxg6 hxg6 Nf3 Nf5 c3 Be7 h3 g5 g4 Nh4 Nxh4 Rxh4 Bd3 Qb6 b4 O-O-O Kf1 Rdh8 Kg2 f6 exf6 gxf6 Be3 Bd6 f3 Bb8 Bf2 R4h6 Qe2 Nf8 c4 Kd7 b5 Qc7 bxc6 bxc6 Be1 Qf4 cxd5 cxd5 Rb1 Ng6 Bxg6 Rxg6 Rb7 Bc7 Qb5 Ke7 Bb4 Kd8 Rxc7 Kxc7 Ba5 Kd6 Qb7 e5 Bb4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d6 Nf3 Nf6 Nc3 Bg4 h3 Bh5 Bd3 e6 Qe2 d5 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 g4 Bg6 Bxg6 hxg6 h4 Na6 O-O-O O-O-O Rdg1 Nc7 Kb1 f6 exf6 Qxf6 Rg3 Rde8 Re1 Rhf8 Nd1 e5 dxe5 Qf4 a3 Ne6 Nc3 Ndc5 b4 Nd7 Qd3 Qf7 b5 Ndc5 Qe3 Qf4 bxc6 bxc6 Rd1 Kc7 Ka1 Qxe3 fxe3 Rf7 Rh3 Ref8 Nd4 Rf2 Rb1 Rg2 Nce2 Rxg4 Nxe6 Nxe6 Nd4 Nxd4 exd4 Rxd4 Rg1 Rc4 Rxg6 Rxc2 Rxg7 Kb6 Rb3 Kc5 Rxa7 Rf1 Rb1 Rff2 Rb4 Rc1 Rb1 Rcc2 Rb4 Rc1 Rb1 Rxb1 Kxb1 Re2 Re7 Rh2 Rh7 Kc4 Rc7 c5 e6 Rxh4 e7 Re4 a4 Kb3 Kc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 g3 Bg4 Bg2 Nd7 h3 Bxf3 Bxf3 c6 d3 e6 e4 Ne5 Bg2 dxe4 Bxe4 Nf6 Bg2 Bb4 Nd2 h5 Qe2 Qc7 c3 Be7 d4 Ng6 h4 e5 Nf3 exd4 Nxd4 O-O-O Bg5 Ng4 O-O-O Rhe8 Qc2 Kb8 Kb1 Bxg5 hxg5 N6e5 Rhe1 c5 Nf3 Rxd1 Rxd1 Nc4 Qa4 Rd8 Re1 Nb6 Qc2 Qd6 c4 Qg6 Qxg6 fxg6 b3 Nxf2 Re6 Kc7 Rxg6 Rd7 Nh4 Nc8 Bd5 Nd6 Re6 Nb5 cxb5 Rxd5 Rg6 Rd7 Nf5 Ne4 Nxg7 Rd1 Kc2 Rd2 Kc1 Rxa2 Nxh5 Nd2 Nf4 Nxb3 Kb1 Rd2 Re6 c4 Re3 Kb6 g6 Kxb5 g7 Kb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 h6 Nxe6 Qe7 O-O fxe6 Bg6 Kd8 Bf4 b5 a4 Bb7 Re1 Nd5 Bg3 Kc8 axb5 cxb5 Qd3 Bc6 Bf5 exf5 Rxe7 Bxe7 c4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 d5 Bg2 dxc4 Ne5 Bb4 Kf1 Bd6 Nxc4 Nd5 e4 Ne7 Bb2 Nbc6 Nd2 O-O Kg1 b5 Nxd6 cxd6 h4 Qb6 h5 h6 d5 Ne5 Nf1 b4 Bd4 Qa5 Ne3 Rac8 Rh4 Rc7 Qd2 Rc3 Bxc3 bxc3 Qd4 exd5 exd5 Qc7 Qd1 Rc8 Be4 Qb6 Rf4 Bb7 Rc1 Qa5 Rc2 Kh8 Bg2 Ba6 Ra4 Qb6 Nc4 Bxc4 bxc4 Nf5 Rxc3 Nd4 c5 Rxc5 Qxd4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 cxd4 Nxd4 Bc5 Qd2 Bxd4 Bxd4 Nxd4 Qxd4 Qb6 Qxb6 Nxb6 a4 Ke7 a5 Nd7 Kd2 g5 g3 gxf4 gxf4 f6 Rg1 fxe5 Bh3 exf4 Rae1 Nf8 Nxd5 Kd6 Nxf4 e5 Rg3 Bxh3 Rd3 Kc6 Rc3 Kd6 Nxh3 Rg8 Rd3 Ke6 Nf4 Kf5 Nd5 Rg2 Kc1 Rf2 Rb3 Rb8 a6 b5 Ne7 Kf6 Nc6 Rb6 Nxe5 h5 Nd3 Rxh2 Nb4 Kg5 Rf3 Ng6 Rf7 Rh4 c3 Rxb4 cxb4 Rxa6 Rg1 Kh6 Kd2 Rd6 Ke3 a6 Ke4 Nh8 Ra7 Ng6 Ra1 Rd2 R7xa6 Rxb2 Kf5 Rf2 Ke6 Rf4 Rb1 h4 Rb6 h3 Rxb5 h2 Rh1 Rh4 Kd6 Nf4 Rf5 Nh5 Rf2 Rxb4 Rfxh2 Rb5 Ra2 Kg6 Ra6 Nf4 Ke7 Kg5 Rg1 Kh4 Ra4 Rf5 Rg8 Kh3 Ra3 Kh2 Ra2 Kh3 Rd2 Re5 Kf6 Re3 Kf5 Ne2 Rb2 Kh2 Rg7 Re8 Rg6 Re3 Re6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 b4 Nh5 Re1 Nf4 Bf1 a5 bxa5 Rxa5 Nd2 c5 a4 Ra6 Ra3 g5 g3 Nh3 Bxh3 Bxh3 Qh5 Qd7 Qxg5 h6 Qe3 f5 Qe2 f4 Nb5 Kh7 gxf4 exf4 Kh1 Bg4 Nf3 Ng6 Rg1 Bxf3 Qxf3 Ne5 Qh5 Qf7 Qh3 Nxc4 Rf3 Be5 Nc7 Rxa4 Bxf4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 dxc4 e3 a6 a4 c5 Bxc4 Nc6 O-O Be7 dxc5 Qxd1 Rxd1 Bxc5 h3 b6 e4 Na5 Bd3 Nb3 Rb1 Nxc1 Rbxc1 Bd7 e5 Nd5 Be4 Nxc3 Rxc3 Rd8 a5 Bb5 Rxd8 Kxd8 Ng5 Kd7 axb6 Bxb6 Rf3 f5 exf6 gxf6 Rxf6 Bd4 Rf7 Kd6 b3 h6 Nf3 Bc3 g4 Rb8 Rh7 Bd7 Rxh6 Rxb3 Ng5 Rb6 Bd3 Bd4 Ne4 Kc7 Rh7 a5 g5 a4 Nf6 Bxf6 gxf6 Rb8 Bc4 Kd6 Rh4 Ke5 Rh7 Kd6 f7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 d6 Bg5 e6 Qd2 a6 O-O-O h6 Be3 Be7 f4 Nxd4 Bxd4 b5 Kb1 Bb7 Bd3 O-O e5 dxe5 fxe5 Nd7 Ne4 Bxe4 Bxe4 Rb8 c3 Qc7 Bc2 Rfd8 Qf4 Nf8 Rhf1 Bg5 Qf2 Rd7 Bc5 Rc8 Bd6 Qd8 Rd4 Qe8 h4 Be7 g4 Bxd6 exd6 Rc6 g5 e5 Rd5 Rcxd6 Rxd6 Rxd6 Bb3 Ne6 g6 f6 h5 Qd7 Qf3 Kf8 Qa8 Nd8 Qe4 Ke7 a3 Qh3 Rd1 Rxd1 Bxd1 Qe6 Bg4 Qc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 c3 e6 d4 d5 exd5 exd5 Nf3 Bd6 dxc5 Bxc5 Bd3 Nf6 O-O O-O h3 Nc6 Nbd2 Bb6 Nb3 Ne4 Nbd4 Qf6 Be3 Re8 Re1 Bd7 Nxc6 bxc6 Bxb6 axb6 Bxe4 dxe4 Qxd7 Red8 Qg4 exf3 Qxf3 Qxf3 gxf3 g6 Re2 Ra5 Kh2 Rd3 a3 Rxf3 Rg1 Raf5 Rg2 Kg7 Rd2 b5 Rc2 R5f4 Rd2 h5 Re2 h4 Rd2 f6 Re2 g5 Rd2 Kg6 Re2 c5 Rd2 c4 Rd5 Rf5 Rd2 Rd3 Re2
Result: 0-1
Processing game...
FEN: 
Moves: c4 Nf6 Nc3 c5 Nf3 e6 g3 b6 Bg2 Bb7 O-O a6 Re1 d6 e4 Qc7 d4 cxd4 Nxd4 Be7 Be3 O-O Rc1 Re8 f4 g6 g4 Bf8 g5 Nfd7 Nde2 Nc6 Ng3 Qb8 a3 Na5 Bf1 Qc7 Nb1 Nc6 Nc3 Na5 b3 Rac8 Bg2 Bg7 Nb1 Qb8 Bf1 Nc6 b4 Na7 Qd2 Red8 Nc3 Nc6 Red1 Bf8 Qf2 Ba8 Rc2 a5 b5 Ne7 Rcd2 d5 f5 gxf5 exf5 e5 Nxd5 Nxd5 cxd5 Bxa3 Ne4 Bb4 Qh4 Bxd2 Rxd2 Nf8 Qh6 Bxd5 Nf6 Kh8 Nh5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 Nc6 Nge2 Nf6 d4 cxd4 Nxd4 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Bg5 g3 Rb8 Bg2 a5 O-O Be6 Nce3 Bxe3 Nxe3 Ne7 Kh1 Qb6 Qe2 b4 c4 Nc6 f4 Nd4 Qd2 f6 b3 Rfd8 Rac1 a4 bxa4 Qa5 Nc2 Nc6 Na1 Qxa4 Nb3 Qa6 Qc2 Ra8 Rf2 Qa7 Rd2 Na5 Qd3 Nb7 h3 Qb6 Qe2 Rdc8 Kh2 Rc6 Bf3 Kh8 Bg4 Bg8 Bd7 Rc7 Bb5 Rac8 Rdc2 Nc5 f5 h6 g4 Qb7 Nxc5 Rxc5 Kg3 Ra8 Rb1 Ra3 Kh2 d5 exd5 Bxd5 Rxb4 Bf3 Qe1 e4 Rb3 Rxb3 axb3 Qb8 Qg3 Re5 Rc1 Qd6 Re1 Qd2 Kg1 e3 c5 Rxc5 Bc4 Rc8 Ra1 Bb7 Re1 Ra8 Be6
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Be3 Be7 c3 O-O Nbd2 Nxd2 Qxd2 Na5 Bc2 Nc4 Qd3 g6 Bh6 Re8 Qd4 f6 exf6 Bxf6 Qf4 Nd6 Nd4 Bd7 Bb3 Nc4 Rae1 Be7 Bxc4 bxc4 Qe5 Bf6 Qxd5 Kh8 Qxc4 Be5 Nc6 Bxc6 Qxc6 Bxh2 Kxh2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Be2 g6 O-O Bg7 Bb5 Nbd7 Re1 a6 Bf1 O-O d4 e5 dxe5 dxe5 a4 Qc7 Na3 Rd8 Qc2 h6 Nd2 b6 Ndc4 Bb7 f3 Bc6 b4 cxb4 cxb4 Rac8 Be3 Qb8 Qf2 Bxa4 Nxb6 Rc6 Nxa4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 d5 O-O O-O c4 dxc4 Na3 c3 bxc3 c5 e3 Nc6 Qe2 Bf5 Bb2 e5 Nxe5 Nxe5 dxe5 Bd3 Qd1 Bxf1 exf6 Qxd1 Rxd1 Bxg2 fxg7 Rfd8 Rxd8 Rxd8 Kxg2 Rd2 Bc1 Rxa2 e4 Kxg7 Nb5 b6 Bf4 a6 Nc7 a5 Nd5 a4 Be5 Kg8 Nxb6 a3 c4 Rc2 Kf3 a2 Ke3 Rc1 Kd2 Rf1 f4 Rf2 Kd3 Rxh2 Nd7 Rh3 Nxc5 Rxg3 Kc2 h5 f5 f6 Bxf6 h4 Kb2 h3 Be5 h2 Bxg3 h1=Q Kxa2 Qg2 Kb3 Qxg3 Kb4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb4 Bg5 Bb7 Nd2 h6 Bh4 O-O e3 d5 cxd5 exd5 Be2 Be7 O-O c5 Bf3 Nbd7 Bg3 Rc8 Nb5 a6 Nd6 Bxd6 Bxd6 Re8 Rc1 cxd4 Rxc8 Qxc8 exd4 Ne4 Bf4 Ndf6 Re1 Nxd2 Rxe8 Qxe8 Qxd2 Ne4 Qb4 Ng5 Bxg5 hxg5 h3 Qe6 Qc3 Qe7 b4 g6 a4 Kg7 a5 Qd6 Be2 b5 Qe3 Qe6 Qxe6 fxe6 Bd3 Kf6 Kf1 Bc6 Ke2 Be8 Ke3 Bd7 g3 Bc8 Bc2 Bd7 Bd1 Bc8 Bg4 Bd7 Be2 Bc8 Bd1 Bd7 Bc2 Bc8 f4 gxf4 Kxf4 g5 Kf3 Kg7 h4 Kh6 Kg4 e5 Kf3 e4 Bxe4 dxe4 Kxe4 gxh4 gxh4 Kh5 Kd5 Kxh4 Kc6 Kg5 Kc7 Be6 Kb7 Kf4 Kxa6 Ke4 Kxb5 Kxd4 a6 Bd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bb4 c3 Bc5 Nxc6 bxc6 Bd3 Qh4 Qe2 Nf6 h3 O-O g3 Qh5 g4 Qe5 g5 Nxe4 Bxe4 Re8 Bf3 Qxe2 Bxe2 Ba6 c4 Bxc4 Nc3 Rab8 Kf1 Be6 b3 f6 gxf6 Bd4 Bb2 Bxf6 Na4 Bh4 Nc5 Rf8 Rh2 Bxh3 Kg1 Rbe8 Nxd7 Bxf2 Rxf2 Bxd7 Bc4 Be6 Rxf8 Kxf8 Rf1 Ke7 Bxe6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O a6 Bb3 b5 a4 b4 Nbd2 Bb7 e4 cxd4 e5 Nd5 Nc4 Nc6 Bg5 Qd7 Rc1 h6 Bh4 Bc5 Nfd2 O-O Ne4 Be7 Bg3 Qd8 Ncd6 Na5 Bc2 b3 Bb1 Qb6 Qd3 g6 Nc5 Bc8 h4 Nc6 a5 Qxa5 Nxf7 Rxf7 Qxg6 Kf8 Nxe6 Bxe6 Rxc6 Bd7 Qxh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 c5 dxc5 O-O a3 Bxc5 Nf3 Nc6 Bg5 Nd4 Nxd4 Bxd4 e3 Qa5 exd4 Qxg5 Qd2 Qxd2 Kxd2 b6 b4 d6 Bd3 Rd8 f3 h5 h4 Kf8 Rhe1 Bd7 Rab1 Rac8 Rec1 e5 Nb5 Bxb5 cxb5 exd4 Rc6 Nd5 Rbc1 Rxc6 bxc6 Nc7 Re1 Re8 f4 Rxe1 Kxe1 Ke7 Be2 g6 f5 Kf6 fxg6 fxg6 Kd2 Kf5 g3 Ke4 a4 Na8 Bd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 Qe2 h6 Ne4 Nxe4 Qxe4 Qc7 Qg4 Rg8 Nd2 Nf6 Qf3 e5 dxe5 Bxe5 Nc4 Be6 Bd2 O-O-O O-O-O Nd7 Rhe1 Rge8 Kb1 g5 h4 Bf4 Bxf4 gxf4 Bf5 Nf8 Qh5 Kb8 Bxe6 Nxe6 a4 Qe7 Qe5 Qc7 Qh5 Qe7 b3 Qf6 Ne5 Re7 Ng4 Rxd1 Rxd1 Qg7 f3 Re8 Qf5 Ka8 h5 Rf8 Rd7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 g3 e5 Nde2 Nbd7 a4 b6 Bg2 Bb7 h3 Be7 g4 Nc5 Ng3 g6 Bh6 Ne6 Qd2 Rc8 O-O-O Rc4 Kb1 Qc7 Rhe1 Rd4 Qc1 Rb4 Bd2 Rd4 Bh6 Rb4 Bd2 Rd4 Bh6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bc5 a4 Bb7 Nc3 O-O d3 Na5 axb5 Nxb3 cxb3 axb5 Rxa8 Bxa8 Nxe5 d5 Bg5 Be7 Nxb5 dxe4 dxe4 Bxe4 Nc3 Bb7 Re1 h6 Bf4 Bb4 Ng4 Nxg4 Qxg4 Re8 Rxe8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O h3 Bb7 d3 d6 c3 Re8 Nbd2 h6 Nf1 Bf8 Ng3 Nb8 Nh4 d5 exd5 Bxd5 Nhf5 Nbd7 Bxd5 Nxd5 Qg4 Kh8 Qf3 N5f6 Ne4 Kh7 h4 Nxe4 dxe4 Qf6 g4 g6 g5 Qe6 Nxh6 Bxh6 gxh6 Nf6 Bg5 Ng4 Rad1 f6 Bc1 Nxh6 Bxh6 Kxh6 Rd3 Rad8 Red1 Rxd3 Rxd3 Rg8 b3 f5 Qg2 fxe4 Qxe4 Rf8 Rg3 Rf4 Qe3 Kh7 Rg5 Qd5 Rxe5 Rg4 Kf1 Qd1 Qe1 Qd3 Qe2 Qb1 Qe1 Qd3 Qe2 Qh3 Ke1 Qxc3 Kf1 Qh3 Ke1 Rg1 Kd2 Qd7 Kc2 Qc6 Kd3 Qd6 Kc2 Qc6 Kd3 Qd6 Kc2 Ra1 Re7 Kg8 Qe6 Qxe6 Rxe6 Rxa2 Kc3 Kg7 Kb4 Kh6 f4 Kh5 Re5 Kxh4 Rg5 Rf2 Rxg6 Rxf4 Ka5 Rf3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Qd2 Nc6 Nb3 Nge5 f3 b5 Bf2 Rb8 Nd4 Bd7 Be2 Nc4 Bxc4 bxc4 Rb1 Rg8 Nxc6 Bxc6 h3 h5 Bd4 Bxd4 Qxd4 Qb6 Qxb6 Rxb6 b4 cxb3 axb3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Nf3 Bb7 Bg5 d6 Nd2 Nbd7 f3 h6 Bh4 Rc8 e4 c5 Bd3 d5 exd5 exd5 O-O cxd4 Qxd4 Nc5 Bxf6 Qxf6 Qxf6 gxf6 Bf5 Rcd8 Rfe1 Ba6 Rac1 dxc4 Nxc4 Nb3 Rc3 Nd4 Bh3 Rfe8 Kf2 Rxe1 Kxe1 Re8 Kd2 Ne2 Re3 Rd8 Kxe2 Bxc4 Ke1 Kf8 Bf5 Re8 Rxe8 Kxe8 Kd2 Ke7 Ke3 Bd5 Kd4 Kd6 Be4 Be6 Bd3 Bc8 Bc4 Ke7 Ke4 Bb7 Bd5 Bc8 Kd4 Bd7 Kc4 Be8 b4 Bd7 f4 f5 Bf3 Kd6 Bh5 f6 Bd1 Bc6 g3 Bd7 Bc2 Ke6 a4 Kd6 a5 Ke6 Kd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 g4 h6 f4 e5 Nf5 h5 gxh5 exf4 Bxf4 Nxh5 Nxd6 Bxd6 Bxd6 Qh4 Kd2 Qg5 Ke1 Qh4 Kd2 Qg5 Ke1 Qh4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c5 bxc5 bxc5 Rb1 Bc6 O-O Nd7 Bb5 Qc7 Qd3 Rfc8 h3 cxd4 Nxd4 Bb7 Rfc1 Nc5 Qd1 Qe7 Qg4 g6 Bf1 Bg7 Qd1 Rab8 Rc2 Ba8 Rbc1 Rd8 g3 Bxd4 exd4 Ne6 h4 Qf6 Ne2 Rb4 Rc8 Bb7 Rxd8 Nxd8 Qd2 Ra4 Qxh6 Rxa2 Qe3 Ne6 Rb1 Bc6 Qh6 Qd8 Qe3 Qf6 Qb3 Rd2 Qe3 Nxd4 Nf4 Qe5 Qa3 Kg7 Rc1 Bb5 Rc8 Be8 Qc3 Bd7 Rd8 Rd1 Rxd7 Qe4 Rxd5 Rxf1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 b5 Bd3 Qb6 Nb3 Qc7 O-O Nf6 Re1 Bd6 g3 b4 Ne2 Nc6 Ned4 Nxd4 Nxd4 h5 Bf1 h4 Bg2 hxg3 hxg3 Bb7 Bd2 Bc5 c3 Qb6 Qe2 d5 e5 Ne4 Be3 Qc7 c4 Qxe5 Nb3 Bxe3 Qxe3 Qh5 Qb6 Rb8 c5 Qh2 Kf1 O-O c6 Ba8 Qxa6 f5 c7 Rbe8 Nd4 Nd2 Ke2 Qxg2 c8=Q Qe4 Kd1 Qxd4 Qaxe6 Rxe6 Qxe6 Rf7 Qe8 Rf8 Qe6 Rf7 Qe8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 f4 Be7 O-O O-O a4 Nc6 Be3 Qc7 Kh1 Re8 Bf3 Bf8 Qd2 Na5 Qf2 Nc4 Bc1 e5 Nde2 d5 fxe5 Nxe5 Bf4 dxe4 Nxe4 Nd5 Bg3 Qe7 Rad1 Nxf3 gxf3 Nf6 N2c3 Nxe4 fxe4 Bh3 Rfe1 Rac8 Bd6 Qe6 Bxf8 Rxf8 Rd3 Qh6 Nd5 Kh8 Rg3 Be6 Reg1 Qh4 Qd4 f6 c4 Rf7 Nb6 Re8 Rd1 h6 Re3 Rfe7 e5 Bg4 Rg3 f5 Nd5 Rxe5 Nf4 Qf6 Rf1 Re1 Qxf6 Rxf1 Kg2 Rxf4 Qc3 Re2 Kg1 Rfe4 h3 Re1 Kh2 R4e2 Rg2 Re3 Qxe1 Rxe1 hxg4 fxg4 a5 h5 Kg3 g5 Rh2 Re3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Nf6 Nc3 d6 f3 Nxd4 Qxd4 Bg7 Be3 O-O Qd2 a5 Be2 a4 O-O Qa5 Rac1 Bd7 c5 Bc6 cxd6 Rfd8 Nb1 Qxd2 Nxd2 exd6 Nc4 d5 Nb6 Ra5 e5 Nd7 f4 Nf8 Bd2 Ra7 Be3 Ra5 Bd2 Ra7 Be3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 d5 Bg2 dxc4 Ne5 Bb4 Kf1 Bd6 Nxc4 Nd5 e4 Ne7 Bb2 Nbc6 Nd2 e5 d5 Nd4 Nf3 c5 Kg1 Bxc4 bxc4 O-O h4 Nc8 Bh3 Re8 Kg2 Bf8 Rc1 Nd6 Re1 Rb8 a4 g6 Nxd4 cxd4 f4 Bg7 fxe5 Bxe5 Bxd4 Rb7 Bf2 Rc7 Qd3 Nb7 Qa3 Bd6 Qb2 Rce7 a5 Nc5 Rcd1 Rxe4 Rxe4 Nxe4 axb6 Nxf2 Qxf2 axb6 Re1 Rxe1 Qxe1 Qc7 Qe3 h5 Qd3 Qe7 Kh1 Qe5 Kh2 Qf6 Kg2 Qb2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Nc3 e5 Bc4 Be7 d3 d6 Nd2 Bg5 h4 Bh6 Qh5 g6 Qd1 Nf6 Nf1 Bxc1 Qxc1 Be6 Ne3 Qd7 Ned5 Bxd5 Nxd5 Nxd5 Bxd5 O-O-O c3 f5 Qe3 Ne7 Bb3 Qc7 h5 d5 Bxd5 Nxd5 exd5 gxh5 c4 Qa5 Kf1 Qb6 Rxh5 Qxb2 Re1 e4 Rxf5 exd3 Qxd3 Qxa2 Rf7 Qa4 g3 Rd7 Ree7 Rxe7 Rxe7 Rf8 Kg2 Kb8 Qxh7 Qb3 Qe4 a5 Re8 Rxe8 Qxe8 Kc7 Qe7 Kb6 d6 Qxc4 d7 Qd5 Kh2 Qh5 Qh4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 g4 h6 Qd2 Be7 O-O-O b5 h4 Nb6 Qf2 Nfd7 Kb1 Rc8 Nd5 Bxd5 exd5 Nc4 Bc1 Qb6 Qe1 a5 Bxc4 bxc4 Nxa5 c3 Nc6 cxb2 Be3 Qa6 Qb4 Rxc6 dxc6 Qxc6 Qb3 O-O g5 Qxf3 gxh6 Rb8 Rhf1 Qxd1 Rxd1 Rxb3 cxb3 gxh6 a4 f5 Bxh6 f4 a5 Kh7 Bg5 Bxg5 hxg5 f3 a6 e4 a7 Nb6 Kxb2 e3 Kc2 e2 Rh1 Kg6 Kd2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6 Qc2 Qa5 Rd1 Be7 Nd2 e5 Bg5 d4 Nb3 Qd8 Be2 a5 Na4 g6 Bxf6 Bxf6 c5 Be6 e4 Qe8 Nb6 a4 Nd2 Ra5 Bd3 Be7 Nxa4 Nb8 b3 Bxb3 Nxb3 Rxa4 Ra1 Qc6 O-O Nd7 Qe2 Nxc5 Rfc1 Ra7 Nxc5 Bxc5 a4 Qd6 Qa2 b6 Qb3 Kg7 Ra2 Ra5 Rd1 h5 Re2 h4 Bb5 Ra7 Qd3 Qe6 Bc4 Qg4 Bb5 Bb4 h3 Qg5 Rc2 Bc3 Qf3 Rc8 Rb1 Rc5 Kh2 Rac7 Bd3 Ba5 Rxc5 Rxc5 Rb5 Rc1 Rd5 Bd2 g3 hxg3 fxg3 Be1
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 e3 Bf5 Nc3 e6 Nh4 Bg4 Qb3 Qb6 h3 Bh5 Bd2 Nbd7 g4 Bg6 g5 Ng8 Nxg6 hxg6 c5 Qc7 e4 Ne7 exd5 exd5 Bd3 Rh4 Ne2 b6 cxb6 axb6 O-O-O b5 Kb1 Nb6 f4 Nc4 Be1 Rh8 Bb4 Nf5 Rhe1 Bxb4 Qxb4 Qd6 Qc3 Kf8 Bxf5 gxf5 Nc1 g6 b3 b4 Qf3 Na3 Kb2 c5 Re5 c4 Ka1 c3 Rxd5 Nc2 Kb1 Na3 Ka1 Qa6 Rd7 Nc2 Kb1 Na3 Ka1 Ra7 Rd8 Kg7 Rxh8 Nc2 Kb1 Na3 Ka1 Kxh8 d5 Nc2 Kb1 Na3 Ka1 Re7 Qd3 Qd6 Qd4 Kg8 Qf6 Nc2 Kb1 Na3 Ka1 Qc5 Nd3 Qe3 Qa6 Qd4 Rc1 Re3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Nd7 d4 Nf6 Nc3 cxd4 Qxd4 e5 Qd3 Be7 Bg5 O-O Bxd7 Nxd7 Bxe7 Qxe7 O-O-O Nf6 Qxd6 Qxd6 Rxd6 Re8 h3 b6 Rhd1 Bb7 Nd2 Rac8 f3 Kf8 Nf1 Ke7 Ne3 g6 a4 Red8 Rxd8 Rxd8 Rxd8 Kxd8 Nc4 Nd7 Nd6 Ba6 Nd5 f6 Kd2 Nc5 b4 Nxa4 b5 Bc8 Nxc8 Kxc8 Nxf6 h5 h4 Kd8 g4 Ke7 Nd5 Ke6 Ke3 Nb2 gxh5 gxh5 f4 Nc4 Kd3 Nb2 Ke2 exf4 Nxf4 Ke5 Nd3 Nxd3 Kxd3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 b5 Bd3 Qb6 Nb3 Qc7 f4 d6 O-O Nd7 Kh1 Ngf6 Qf3 Bb7 a3 g6 Qh3 Bg7 f5 gxf5 exf5 e5 Bg5 h6 Bh4 Rg8 Rae1 Bf8 Re2 Rg4 Nd2 h5 Nde4 Be7 Bxf6 Nxf6 Nxf6 Bxf6 Be4 O-O-O Qxh5 Rh4 Bxb7 Qxb7 Qf3 Qxf3 Rxf3 d5 Rd3 Rf4 g3 Rxf5 Rxd5 Rxd5 Nxd5 Bd8 Kg2 Kd7 h4 Ke6 g4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d5 Nf3 dxc4 Na3 c3 bxc3 O-O O-O c5 e3 Nc6 Qe2 Qa5 Bb2 Nd5 Rfc1 Bg4 h3 Bxf3 Bxf3 Nb6 Rab1 cxd4 cxd4 e5 Qb5 Qxb5 Nxb5 exd4 Bxc6 bxc6 Nxd4 Na4 Ba3 c5 Rc4 Nb6 Rxc5 Bxd4 exd4 Rfd8 Rd1 Rd7 Bc1 Rad8 Be3 f6 Kg2 Kf7 g4 Nd5 Kf3 Nc7 Ra5 a6 h4 Rd5 Rxd5 Rxd5 Rc1 Ne6 Ke4 Ra5 d5 Ra4 Kf3 Nd4 Bxd4 Rxd4 Rc7 Ke8 Rxh7 Rxd5 Rg7 Ra5 Rxg6 Kf7 Rh6 Kg7 Rh5 Ra3 Kg2 Rxa2 Rc5 a5 Kg3 a4 Ra5 a3 Ra7 Kg6 h5 Kh6 Ra6 Kg7 h6 Kxh6 Rxf6 Kg7 Ra6 Ra1 Kf4 a2 Kf5 Kf7 g5 Kg7 Ra7 Kg8 Kf6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Bxf6 gxf6 Nd5 f5 c3 Bg7 exf5 Bxf5 Nc2 O-O Nce3 Be6 Bd3 f5 O-O Ra7 Qh5 Raf7 g4 e4 Bc2 Ne5 Nf4 Bd7 Bb3 fxg4 Ne6 Bxe6 Bxe6 Qf6 Bxf7 Rxf7 h3 Qe6 hxg4 Rf6 Kg2 Nd3 Qh4 Nf4 Kg1 h5 g5 Nh3 Kh1 Rf3 Qxh5 Nxf2 Kg2 Rh3 Qe2 Qe5 Rg1 Rh2 Kf1 Nh3 Qg4 Nxg1 Kxg1 Rxb2 Qc8 Bf8 Qh3 Qxg5 Kh1 Qh6 Rg1 Kh7 Qxh6 Bxh6 Nd5 Bg7 Rg2 Rxg2 Kxg2 Kg6 Kf2 Kf5 Ke3 a5 Ne7 Ke6 Nc6 d5
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 Nf3 d6 Nd2 Nbd7 f3 h6 Bh4 c5 e4 cxd4 Qxd4 Qc7 Be2 Rac8 O-O Qc5 Bf2 Qxd4 Bxd4 Rc7 b4 Rfc8 Rfd1 Ne8 Nf1 Kf8 Ne3 Ke7 Kf2 g6 h4 h5 Rd2 a5 Rb2 Ra8 Rbb1 axb4 axb4 Rcc8 Ke1 Nc7 Kd2 e5 Bc3 Rxa1 Rxa1 Ra8 Rxa8 Bxa8 g3 Bc6 Bd3 f6 f4 exf4 gxf4 Ne6 f5 Nf4 fxg6 Nxg6 Nf5 Ke6 Nd4 Kf7 Nxc6 Nxh4 Be2 Kg6 Ne7 Kg5 Bd4 Ng2 Bf1 Nh4 Ke3 Ng6 Nxg6 Kxg6 Bh3 Ne5 Be6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nf6 Nxf6 exf6 c3 Be7 Bd3 Be6 Ne2 O-O Qc2 g6 h4 f5 h5 Re8 Nf4 Bf6 Kf1 Nd7 Bd2 Nb6 Re1 Bc4 hxg6 hxg6 b3 Bxd3 Qxd3 Rxe1 Bxe1 Nd5 Bd2 Nxf4 Bxf4 Qe7 Qh3 Kf8 Qh7 Ke8 Rh3 Kd7 Re3 Qf8 Be5 Bxe5 dxe5 Re8 Qh4 Kc8 Qd4 a6 Rd3 Rd8 Qxd8 Qxd8 Rxd8 Kxd8 Ke2 Kd7 Kd3 Ke6 Kd4 b6 b4 f6 exf6 Kxf6 f4 Ke6 c4 Kd6 c5 bxc5 bxc5 Ke6 Kc4 a5 g3 Kf6 Kb3 Ke6 Kc4 Kf6 Kd4 Ke6 Kc4 Kf6 Kd4 Ke6 Kc4 Kf6 Kd4 Ke6 Kc4 Kf6 Kd4 Ke6 Kc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c3 c5 Bg5 h6 Bxf6 Qxf6 e4 cxd4 cxd4 Bb4 Nc3 O-O Rc1 Nc6 a3 Ba5 b4 Bb6 e5 Qd8 Ne4 d5 Nc5 f6 Be2 fxe5 dxe5 a5 Qd2 axb4 axb4 Bc7 Nd3 Ne7 O-O Nf5 Rfe1 Bb6 Nc5 Qe7 Nd4 Nxd4 Qxd4 Bd7 Bd3 Ra3 Bb1 Bb5 Qg4 Qf7 f3 Bd3 Bxd3 Rxd3 Kh1 Bxc5 Rxc5 Qf5 h3 Rb3 Rb5 Qxg4 hxg4 Rf4 Rxb7 Rfxb4 Re7 Rb1 Rxb1 Rxb1 Kh2 Rb6 Kg3 Kf8 Rd7 Rb2 f4 Re2 Kf3 Re4 g3 g6 Rh7 Ra4 Rxh6 Ra3 Kg2 Ra2 Kf3 Ra3 Kg2 Ra2 Kh3 Kg7 g5 d4 Kg4 d3 Rh1 Rf2 Ra1 d2 Ra7 Kf8 Rd7 Ke8 Rd6 Ke7 Kh3 Kf7 Rd3 Ke7 Rd4 Ke8 Rd6 Ke7 Kh4 Rh2 Kg4 Rf2 Kh3 Kf7 Rd7 Ke8 Rd4 Ke7 Kg4 Kf7 Kh3 Ke7 Kh4 Rh2 Kg4 Rf2 Kh3 Kf7 g4 Ke7 Kg3 Re2 Kf3 Rh2 Ke3 Rg2 Kf3 Rh2 Rd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nf6 Nxf6 exf6 c3 Be6 Bd3 Bd6 Ne2 O-O Bf4 Re8 O-O Bg4 f3 Bh5 Qd2 Bg6 Bxd6 Qxd6 Nf4 Nd7 Nxg6 hxg6 Rfe1 f5 g3 c5 Bb5 Rxe1 Rxe1 Nf6 Qe3 cxd4 Qxd4 Qxd4 cxd4 Kf8 Kf2 Rd8 Rd1 Nd5 Bc4 Nb6 Bb3 Rd7 h4 Ke7 Ke3 Kd6 Kd3 a5 a3 Re7 Rc1 Na8 Rc5 b6 Rd5 Kc6 Re5 Ra7 Re8 Nc7 Re7 Ra8 Rxf7 Rd8 Rxg7 Rd6 Bf7 Rd7 Rxg6 Kb7 Rf6 Nb5 Be8 Rxd4 Ke3 Rd6 Rxd6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Bxc6 dxc6 d3 Nd7 b3 c5 Bb2 Bd6 Nbd2 O-O Nc4 f6 Nh4 Nb8 Nf5 Bxf5 exf5 Nc6 Qf3 Qd7 Qe4 Kh8 Rfe1 b5 Ne3 Nd4 c3 Nc6 Rad1 Rad8 h4 Rfe8 h5 h6 g3 Ne7 g4 c6 c4 Bb8 Ng2 Qb7 Nh4 Rd7 Ng6 Kg8 f4 Nxg6 fxg6 Rde7 f5 Rd7 Bc3 Red8 Kf1 Bc7 Re3 Bb6 Rf3 Qc7 Kg2 Rd6 Rh3 R8d7 g5 Ba5 Bb2 Qd8 gxh6 gxh6 Qe3 bxc4 bxc4 Qb8 Ba3 Bb4 Bc1 Bc3 Qe2 Kg7 Qc2 Qb4 Kh1 Qa5 Bxh6 Kxh6 Qc1 Kg7 h6 Kh8 g7 Kh7 Rg1 Rd8 g8=Q Rxg8 Rg7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 g4 b5 Qd2 Nb6 Na5 b4 Ne2 d5 g5 Nfd7 c3 bxc3 bxc3 Bc5 Bxc5 Nxc5 Qe3 Qc7 Ng3 O-O Nb3 Nca4 Rc1 Rfd8 Be2 Nc4 Qf2 Qe7 h4 Qa3 O-O Qxa2 Bxc4 Qxf2 Rxf2 dxc4 Na5 Rd3 Ne2 Rb8 Nc6 Rb5 Nb4 Rd6 f4 exf4 Nxf4 Nxc3 e5 Rd4 Nxe6 Rg4 Kf1 Rxb4 Nd8 Nd5 Rxf7 c3 Rd7 Rbf4 Ke2 Rg2 Kd3 Rd2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Bf4 O-O Rc1 dxc4 e4 b5 Nxb5 Nxe4 Bxc7 Qd7 Bxc4 Nc6 d5 Nb4 Be5 Bh6 O-O Ba6 Nc7 Bxc1 Qxc1 Nd3 Bxd3 Bxd3 Rd1 Be2 Re1 Bxf3 gxf3 Nd6 Nxa8 Rxa8 Qc6 Qd8 Qa4 Nf5 Qe4 Rc8 Bc3 h5 Qe5 Kh7 Rd1 Rc4 Kf1 Rc8 a4 Qd7 a5 a6 Re1 Rd8 Rd1 f6 Qe4 Qb5 Kg2 e5 f4 Nh4 Kh1 Qb3 Re1 Qxd5 Qxd5 Rxd5 fxe5 fxe5 Rxe5 Rd7 f4 Kg8 Re3 Rd1 Re1 Rd5 Rb1 Kf7 b4 Nf5 b5 axb5 a6 Ne7 a7 Rd8 Rxb5 Ra8 Bd4 Nc6 Be3 Ke6 Rb7 Kd6 Kg2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7 Bf2 Nc6 Qd2 Ne5 b3 e6 Nde2 Rc8 Bd4 b5 Nd1 Rg8 Ne3 a5 O-O-O a4 Kb1 axb3 cxb3 Ra8 Bc3 Ra6 Nc2 Bf8 Nb4 Ra8 Nd4 Be7 Rhe1 Qc8 Bb2 Qb7 Ndc2 f6 Nd4 h4 g3 g4 f4 Nf3 Nxf3 gxf3 f5 hxg3 hxg3 Rxg3 Qh2 Rg5 Bc1 Rg7 Qh5 Rf7 Qxf3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bg5 e6 Qd2 a6 O-O-O Be7 f4 Nxd4 Qxd4 b5 Bxf6 gxf6 e5 d5 Kb1 Bb7 f5 fxe5 Qxe5 Bf6 Qg3 Qe7 fxe6 fxe6 Be2 O-O-O Bg4 h5 Bh3 h4 Qf4 Bg7 Rhe1 Rh6 a3 Rg6 Bxe6 Rxe6 Qg4 Rd6 Rxe6 Qxe6 Qxg7 Qf6 Qg4 Kb8 Rd4 Re6 Qxh4 Qf1 Ka2 Qxg2 Rg4 Qf1 Rg8 Ka7 Qd4 Rb6 Nxd5
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 Nc6 g3 g6 Bg2 Bg7 Nc3 O-O Nc2 d6 O-O Be6 b3 Qd7 Bb2 Bh3 Ne3 Bxg2 Kxg2 Rfd8 Qd2 e6 Rfd1 d5 cxd5 exd5 Ncxd5 Nxd5 Nxd5 Qxd5 Qxd5 Rxd5 Rxd5 Bxb2 Rad1 Kf8 e3 Bf6 Rd7 Rb8 f4 Ke8 Kf3 Rd8 Rxd8 Bxd8 e4 f6 h4 Bb6 h5 Nd4 Kg4 Kf7 hxg6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 c5 cxd5 g5 Bf2 Nxd5 Qd2 cxd4 exd4 e5 h4 exd4 hxg5 Re8 Ne2 d3 Rxh6 dxe2 Bd4 f6 gxf6 exf1=Q Kxf1 Ne3 Kf2 Nc6 Bxe3 Qxd2 Bxd2 Ba6 Bc3 Re2 Kg3 Kf7 Rh7 Ke6 Rah1 Bc4 R1h6 Rg8 Kf4 Rf8 g4 Bd5 g5 Rf2 Rh3 Rg2 Rg3 Rxg3 Kxg3 Kf5 Rg7 Bf7 Kf2 a5 Ke3 Nd8 g6 Bxg6 Rd7 Bf7 Rd6 Re8 Kf2 Ne6 Rxb6 a4 Ra6 Nf4 Ra5 Kg6 Rxa4 Nd3 Kg3 Be6 Ra7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 Bf4 e5 Be3 a6 N5c3 Nf6 Bg5 Be7 Bxf6 Bxf6 Nd2 O-O Nc4 b5 Qxd6 bxc4 Qxc6 Be6 Be2 Rb8 O-O Qd2 Rab1 Qxc2 Rfc1 Qd2 Rd1 Qc2 Qxa6 Rfc8 h3 Be7 Rdc1 Qd2 Rd1 Qc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 g4 h6 Qd2 b5 O-O-O Nb6 Qf2 Nfd7 Kb1 g6 h4 h5 Nd5 Bxd5 exd5 Qc7 f4 Nc4 gxh5 Rxh5 Be2 Rf5 h5 gxh5 Bc1 Nf6 Bd3 e4 Qe2 Qe7 Bxc4 bxc4 Qxc4 Qd7 Rhe1 Be7 Rg1 Qb5 Qc3 Rxd5 Rg5 Rxg5 fxg5 Nd7 Qh8 Nf8 Qxh5 Ng6 Qg4 Qe5 Nd4 d5 Be3 Rb8 b3 Ba3 Rf1 a5 Qh3 a4 Rf5 Qd6 Qh7 Rb7 Rf1 axb3 axb3 Qe5 Qh3 Ne7 Qh7 Ng6 Qh3 Ne7 Rf6 Bc5 Nc6 Nxc6 Bxc5 Ne7 Qh8 Kd7 Rd6 Qxd6 Bxd6 Kxd6 Qf6 Kd7 Qxf7 Rb6 Kc1 Rg6 Qf4 Re6 Kd1 e3 Ke2 Kc6 c4 dxc4 Qxc4 Kd7 b4 Rg6 Qg4 Kc7 Kxe3 Nd5 Kf2 Kd6 b5 Kc5 Qf5 Rg7 g6 Kd6 b6 Rb7 Kg3 Rg7 Kh4 Rg8 b7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Nbd7 Bg5 c6 e3 Qa5 Nd2 Bb4 Qc2 O-O Be2 c5 Nb3 Qc7 O-O dxc4 Bxc4 cxd4 Nb5 Qb8 N5xd4 Bd6 f4 h6 Bh4 a6 Be2 Nd5 Qd2 N7f6 Rad1 Bb4 Qd3 Bd7 Bf3 Ba4 Kh1 Be7 Rc1 Rc8 Na5 Rxc1 Rxc1 Qd8 Nxb7 Qb6 b3 Bb5 Nxb5 axb5 Bg3 Nh5 Bh4 Ng3 Bxg3 Bf6 Bxd5 exd5 Nc5 Bb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 a6 h4 Qb6 Na4 Qa5 Nc3 cxd4 Nxd4 Bb4 Nb3 Bxc3 bxc3 Qxc3 Kf2 Nb4 Bd3 O-O Kg1 f6 Bd2 Qc7 Bxh7 Kxh7 Bxb4 Rf7 Bd6 Qc4 Qd3 Kg8 Nd4 fxe5 fxe5 Nb6 h5 Na4 Rh4 Bd7 h6 Qxd3 cxd3 Rc8 hxg7 Rxg7 Rf1 Nc3 Rf6 Nb5 Nxe6 Bxe6 Rxe6 Rc2 Re8 Kf7 e6 Kf6 Rf4 Kg5 Be5 Rg6 e7 Re2 Ref8 Rxe5 R4f5 Rxf5 e8=Q Rxf8 Qxf8 Nd6 Qf3 b5 g3 Re6 Qxd5 Kf6 g4 Nf7 Qf5 Ke7 d4 Rf6 Qe4 Kf8 Kg2 Kg7 Kg3 Rg6 d5 Rd6 Kf4 Rf6 Ke3 Rd6 Kd4 Rg6 Kc5 Rf6 Qd4 Kg6 Qe4 Kg5 Qe7 Kg6 a3 Kg5 Kb4 Kg6 Kc5 Kg5 Qe4 Kh6 Qh1 Kg7 Qh5 Rh6 Qf5 Rf6 Qe4 Kh6 Qe3 Kg6 Qh3 Kg7 g5 Nxg5 Qg4 Rg6 Kb4 Nf7 Qd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6 Rc1 a6 cxd5 exd5 Bd3 Ba7 O-O d4 exd4 Nxd4 Nxd4 Qxd4 Be3 Qh4 Bxa7 Rxa7 Qa4 Qxa4 Nxa4 b5 Nc5 Nd7 Rfd1 Nxc5 Rxc5 g6 f3 Be6 Kf2 b4 a4 Bb3 Bc2 Bxc2 Rxc2 b3 Rc3 Rb8 Rd4 h5 h4 Kg7 Rc6 Re7 Rxa6 Rc7 Rd2 Rc2 Rd6 Rb4 a5 Rxd2 Rxd2 Rxh4 g4 hxg4 Kg3 Rh5 fxg4 Rxa5 Rd3 Rb5 Rf3 f5 gxf5 gxf5 Rc3 Kh6 Kf3 Kg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 Nf3 Nf6 c4 dxc4 Nc3 a6 e4 b5 e5 Nd5 a4 e6 axb5 Bb4 Qc2 Nb6 Qe4 Qd5 Qg4 axb5 Rxa8 Qxa8 Be2 Rg8 O-O Bxc3 bxc3 Nc6 Bd1 Ne7 Bc2 Nf5 Ng5 h6 Ne4 Kf8 Nc5 Nd5 Qf3 Nfe7 Bh7 Rh8 Be4 Qa2 h4 g6 Bb1 Qa5 Bd2 Kg7 h5 Qa3 Rc1 b4 cxb4 Qxf3 gxf3 Rd8 hxg6 Nxg6 Be4 c3 Bxc3 Nxc3 Rxc3 Rxd4 Ra3 Rxb4 Ra7 Nxe5 Rxc7 Rb8 f4 Nc4 Bf3 Nd6 Rc6 Nf5 Rc7 Nd6 Rc6 Ne8 Kh2 Nf6 f5 exf5 Kg3 f4 Kxf4 Rb4 Ke3 Bf5 Rd6 h5 Rd4 Rb2 Nd3 Rb3 Be2 Ng4 Kf4 Bxd3 Rxd3 Rb4 Kg5 Nxf2 Rf3 Ne4 Kxh5 f5 Rxf5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 e3 b5 a4 b4 Nb1 Ba6 Qc2 b3 Qd1 e6 Be2 c5 O-O Nc6 Ne5 Rc8 Na3 cxd4 Naxc4 Qd5 Bf3 Ne4 Nxc6 Rxc6 Nd2 f5 Re1 d3 Nxb3 e5 Bd2 Rb6 Bxe4 fxe4 Nc1 Rxb2 Bc3 d2 Rf1 Bxf1 Bxb2 Bc4 Ne2 Bb3 Nc3 Bxd1 Nxd5 Bb3 Nc3 Bb4 Nd1 O-O
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 c5 e3 d6 Bd3 cxd4 exd4 d5 cxd5 Qxd5 Bxf6 gxf6 Ne2 Rd8 Ng3 Qxd4 Be4 Qxc3 bxc3 Rd5 Rd1 Bb7 Bxd5 Bxd5 Nh5 Nd7 c4 Bc6 Rxd7 Bxd7 Nxf6 Kg7 Nxd7 f6 Ke2 Rc8 Rc1 Rc7 Nb8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5 O-O Nd7 b4 c5 bxc5 bxc5 Rb1 Bc6 Bb5 Qc7 Qd3 Rfc8 Rfd1 Rab8 a4 cxd4 Nxd4 Nc5 Qf5 Bxd4 exd4 g6 Qf3 Ne4 Nxe4 dxe4 Qe3 Qd7 d5 Bxb5 axb5 Rxb5 Rxb5 Qxb5 h4 Qa4 Re1 Rd8 h5 g5 Qc5 Qa2 Rd1 Qe2 Qd4 Rb8 Qd2 Qxd2 Rxd2 Ra8 Rd4 f5 g4 Kf7 gxf5 Kf6 Rxe4 Kxf5 Re7 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 c5 dxc5 bxc5 e3 d6 O-O-O Ne4 Qd3 Nxf2 Bxd8 Nxd3 Bxd3 Rxd8 Be4 d5 cxd5 Bb7 Ne2 exd5 Nc3 Nc6 Bxd5 Na5 Bxb7 Nxb7 Rhf1 Kf8 Kc2 Rac8 Rxd8 Rxd8 Rf5 Ke7 Ne4 Rc8 Rg5 Kf8 Rd5 Ke7 Re5 Kd7 g4 f6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 c6 Bg5 Nbd7 e3 Qa5 Nd2 Bb4 Qc2 O-O a3 dxc4 Bxf6 Nxf6 Nxc4 Bxc3 bxc3 Qd5 Bd3 b5 e4 Qg5 Ne5 Qxg2 O-O-O Qg5 Qd2 Qxd2 Kxd2 Bb7 Ke3 c5 Rb1 a6 a4 Rfc8 axb5 axb5 Rxb5 cxd4 cxd4 Ba6 Rc5 Bxd3 Rxc8 Rxc8 Kxd3 Ra8 Rb1 Ra3 Kc4 Ra4 Kd3 Ra3 Kc4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4 Re1 Nxd3 Qxd3 Nf6 c5 O-O Nc3 c6 Bf4 Bg4 Rxe7 Qxe7 Bd6 Qd8 Bxf8 Qxf8 Ne5 Re8 b4 Bc8 b5 Nd7 Nf3 f6 a4 Qe7 a5 a6 bxc6 bxc6 Na4 Qe4 Qxe4 dxe4 Nd2 Nf8 Nc4 Be6 Nd6 Rb8 Nb6 f5 Re1 g6 f3 exf3 gxf3 Kg7 f4 Rd8 Kf2 Kf6 h4 h6 Re5 g5 hxg5 hxg5 fxg5 Kxg5 Rxf5 Bxf5 Nf7 Kf4 Nxd8 Ke4 Nxc6 Ne6 Nb4 Kxd4 c6 Nc7 Ke2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3 Qxc3 dxc4 Qxc4 b6 Nf3 Ba6 Qc2 Nbd7 Bg5 O-O e4 Bxf1 Rxf1 h6 Bxf6 Nxf6 O-O-O Rc8 Kb1 Qe7 Rfe1 Rfd8 Ne5 c5 d5 exd5 exd5 Qc7 Nc6 Rd7 Ne7 Rxe7 d6 Qxd6 Rxd6 Rxe1 Rd1 Re7 f3 Rce8 Rd2 Nd7 Ka2 Nf8 Qa4 Rc8 h4 c4 Rc2 Rec7 Rc3 Ne6 Qd1 b5 Qd6 Rd8 Qa6 Rd5 Re3 Kh7 g4 Rd2 Rc3 Nc5 Qxb5 Rb7 Qxb7 Nxb7 Rxc4 Rd7 f4 h5 gxh5 Kh6 Ra4 Nd6 Ra5 Ne4 Re5 Nf6 b4 Nxh5 f5 f6 Rc5 Ng3 b5 Kh5 a4 Kxh4 a5 Kg4 Kb3 Nxf5 Kb4 g5 b6 axb6 axb6 Ne7 Rb5 Rb7 Ka5 Nc8 Rb4 Kh5
Result: 0-1
Processing game...
FEN: 
Moves: e4 e6 d3 d5 Nd2 c5 g3 Nc6 Bg2 Nf6 Ngf3 Be7 O-O Qc7 Re1 b6 Nf1 dxe4 dxe4 Ba6 e5 Rd8 Bd2 Ng4 c3 O-O Qa4 Bxf1 Qxg4 Bxg2 Bh6 g6 Kxg2 Rfe8 Re4 Rd3 Rae1 Bd8 h4 Ne7 h5 Nf5 Bg5 Qd7 Bf6 Rd1 Rxd1 Qxd1 Re1 Qd3 hxg6 fxg6 Qa4 Rf8 Qxa7 Bxf6 exf6 Rxf6 Qb8 Rf8 Qxb6 Ng7 Re3 Qd5 c4 Qxc4 Qb3 Qg4 a4 Qf5 a5 c4 Qxc4 Qxa5 b4 Qb6 Ng1 Qd8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3 Qxc3 Ne4 Qc2 e5 cxd5 Qxd5 f3 Nf6 e4 Qxd4 Qxc7 Nc6 Ne2 Qc5 b4 Nxb4 Qxc5 Nd3 Kd2 Nxc5 Rb1 O-O Ke3 Be6 Bb2 Rfd8 Bxe5 Rd3 Kf2 Ncxe4 Kg1 Nd2 Ra1 Nxf1 Nf4 Rd2 Rxf1 Bc4 Rc1 Nd7 Bc3 Ra2 h4 b5 Rh3 f6 Rd1 Ne5 Rg3 Bf7 Rd6 Rxa3 Bd4 Ra4 h5 h6 Bxe5 fxe5 Nd3 Rd4 Rxd4 exd4 Rg4 a5 Rxd4 a4 Rb4 a3 Nc1 Bc4 Rb1 a2
Result: 0-1
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nb5 Nf6 Bf4 Bb4 N1c3 O-O Bd6 Bxd6 Nxd6 Qb6 Qd2 Ne8 Nxe8 Rxe8 e3 Rd8 Rc1 d6 Be2 a6 O-O Qc7 Rfd1 Qe7 Na4 Qc7 Nc3 Qe7 e4 b6 Qe3 Rb8 a3 Bd7 b4 Be8 h3 h6 Rd2 Rdc8 f4 Qc7 Rcd1 Rd8 Kh1 Ne7 Rd3 b5 c5 dxc5 Rxd8 Rxd8 Rxd8 Qxd8 bxc5 Bc6 Bf3 Qa5 Ne2 Qe1 Kh2 Qa5 Nd4 Qc7 Kh1 Bd7 Nb3 a5 Qc3 Nc6 e5 b4 axb4 Nxb4 c6 Be8 Qc5 Nxc6 Nd4 Qa7
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7 b4 a5 Ba3 axb4 Bxb4 Nd7 a4 f5 Ng5 Nc5 Bxc5 dxc5 Bf3 Ra6 a5 Kh8 Ne6 Bxe6 dxe6 f4 Qxd8 Rxd8 Rfb1 Rb8 Nd5 Nxd5 cxd5 Bf8 Be2 Ra7 a6 Bd6 h4 Kg7 Bc4 b6 Bb5 h6 Kf1 Rf8 Ke2 Rf6 Rb3 Kf8 Rh3 Ke7 Kd3 Kf8 Kc4 Kg7 Bc6 Kf8 Kb5 Kg7 Bb7 Kf8 Rg1 Ke7 g3 f3 Rc1 Kd8 Rc3 h5 Rh1 Ke7 Rd1
Result: 1-0
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 g3 d6 Bg2 Bd7 Nc3 g6 Nxc6 bxc6 c5 d5 e4 dxe4 Nxe4 Nxe4 Bxe4 Bg7 O-O O-O Qa4 Qc8 Rd1 a5 Bg5 Rb8 Qxa5 Rxb2 Rab1 Rxb1 Rxb1 Bf5 Qe1 Qe6 Bxf5 Qxf5 Bxe7 Re8 Rd1 Bf8 Rd8 Rxd8 Bxd8 Qxc5 Qe2 Qd5 Bf6 c5 Bc3 c4 Qe5 Qxe5 Bxe5 Bb4 Kf1 f5 Ke2 Kf7 Bf4 Ke6 Bd2 Ba3 Bc3 Kd5 f3 h5 h3 Bd6 g4 hxg4 hxg4 fxg4 fxg4 Ke4 a4 Bc7 a5 Bxa5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Nc3 e6 e3 Nbd7 Qc2 Bd6 b3 O-O Be2 b6 O-O Bb7 Bb2 c5 cxd5 exd5 Rad1 Rc8 dxc5 bxc5 Qf5 Qe7 Rfe1 Qe6 Qb1 h6 Bd3 Ne5 Bf5 Nxf3 gxf3 Qe5 f4 Qe8 Bh3 Rd8 Bg2 Qe6 Ne2 Bc7 Ng3 Ba5 Re2 Ba6 Rc2 Bb6 Rcd2 d4 exd4 cxd4 Bxd4 Bxd4 Rxd4 Rde8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 d5 e3 c6 Bd3 Bg4 f3 Bh5 Ne2 e6 Nf4 Bg6 Nxg6 hxg6 c4 dxc4 Bxf6 Qxf6 Bxc4 Bd6 Nc3 Qh4 Ke2 Nd7 Ne4 Qe7 a3 Bc7 b4 Nb6 Bb3 Nd5 f4 f5 Bxd5 exd5 Nc5 Bxf4 Qg1 O-O Nd3 Bd6 Qf2 f4 Kd2 fxe3 Qxe3 Qxe3 Kxe3 Rae8 Kd2 Re4 Kc3 Rfe8 Rae1 Rxe1 Rxe1 Rxe1 Nxe1 Bxh2 Nd3 Bd6 a4 Kf7 Kd2 Ke6 Ke3 Kf5 Kf3 Kg5 a5 Kf5 Nc5 b6 axb6 axb6 Nd3 Kg5 Ke3 Kg4 Kf2 b5 Ke2 Kg3 Kf1 Be7 Kg1 Bg5 Kf1 Be3 Ne5 Bxd4 Nxc6 Bc3 Na7 Bxb4 Nxb5 Ba5 Nd4 Kf4 Ke2 Ke4 Nf3 Bd8 Ne1 Bf6 Nd3 g5 Nf2 Kd4 Nd3 Kc3 Nf2 Be5 Nd3 Bd4 Ne1 g4 Nd3 Bf6 Nf4 Kc4 Nd3 Kd4 Nf2 g3 Kd2 Be7 Nd3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Nc3 e6 e3 Nbd7 Qc2 Bd6 b3 O-O Be2 b6 O-O Bb7 Bb2 c5 cxd5 exd5 Rad1 Rc8 dxc5 bxc5 Nh4 g6 Nf3 Qe7 Rfe1 a6 h3 Rfd8 Bf1 Bb8 Ne2 Ne4 Nf4 Ndf6 g3 d4 Bg2 Qf8 Qc1 Re8 Qa1 Bxf4 exf4 a5 a3 Bd5 b4 axb4 axb4 Bb3 Rc1 Nd5 bxc5 Ra8 Qb1 Nec3 Bxc3 Ba2 Qd3 dxc3 Ne5 Qxc5 Bxd5 Bxd5 Rxc3 Qa5 Rec1 Be6 Ng4 Bxg4 hxg4 Rad8 Qf3 Qd5 Kg2 Qxf3 Kxf3 h6 Rc6 Kg7 R1c3 Rd5 Re3 Red8 Rcc3 R8d7 Kg2 Ra7 Re5 Rad7 Rce3 R5d6 f5 g5 Ra5 Kf6 Rea3 Kg7 Ra1 Rd4 Kh3 Rd2 Kg2 R2d4 R5a4 Rxa4 Rxa4 Rd3 Ra6 Rb3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 g3 g6 Bg2 Bg7 Nc3 O-O O-O Ng4 e3 d6 Nde2 Qa5 Nd5 Qc5 Bd2 e6 b4 Qxc4 Rc1 Qxa2 Nec3 Qc4 Nc7 Rb8 N3d5 Qa2 Nc3 Qc4 N3d5 Qa2 Nc3 Qc4 Nb1 Nxf2 Rxf2 Qd3 b5 Ne5 Rc3 Bd7 Rxd3 Nxd3 Rf1 Rfc8 Bc3
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Qb3 e6 Bg5 Nbd7 Nc3 h6 Bh4 Be7 e3 O-O Rd1 Ne4 Bxe7 Qxe7 Bd3 e5 dxe5 Ndc5 Qc2 Nxd3 Qxd3 Nxc3 Qxc3 dxc4 Rd6 Be6 O-O Rfd8 Rfd1 Bd5 Rxd8 Rxd8 h3 b5 Nd4 b4 Qd2 Re8 f4 Be4 Qe2 Rd8 Rc1 c5 Nf3 Bd3 Qe1 a5 e4 Qb7 Nd2 a4 Nxc4 Qxe4 Qxe4 Bxe4 Nd6 Bd5 a3 bxa3 bxa3 c4 Rc2 f6 Nxc4
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Be2 Bb7 O-O a6 e4 c5 d5 exd5 e5 Ne4 Nxd5 Nxe5 Nxe5 Qxd5 Qxd5 Bxd5 Rd1 Rd8 Bf3 f5 Be3 Be7 Nd3 c4 Nf4 Bf7 Bb6 Rxd1 Rxd1 O-O Nd5 Bxd5 Rxd5 g6 a4 bxa4 Be2 Rc8 Bd4 a3 f3 c3 bxc3 Nxc3 Bxa6 Nxd5 Bxc8 Kf7 Ba6 Ke6 Bc4 Kd6 Kf1 Bf6 Bf2 Nb4 Be1 Kc5 Bg8 h6 Ke2 Nc2 Bd2 Nd4 Kd3 Nc6 Kc2 Nb4 Kb1 a2 Bxa2 Nxa2 Kxa2 h5 Kb3 Kd4 Kc2 Kd5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e6 g3 Bb4 Nc3 O-O Bg2 d5 cxd5 Nxd5 Qb3 Qb6 Bxd5 exd5 Be3 Bxc3 Qxc3 Qg6 O-O Nc6 Rfc1 Bh3 Qc2 Nxd4 Bxd4 Qe6 f3 Rfe8 Kf2 Qh6 Qd3 Re7 Rc2 Rae8 Re1 Qh5 Kg1 Bf5
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 e3 e6 Bd3 dxc4 Bxc4 c5 O-O Nc6 Qe2 a6 dxc5 Bxc5 a3 b5 Bd3 Bb7 b4 Be7 Bb2 O-O Nbd2 Qb6 Nb3 a5 bxa5 Nxa5 Bd4 Qd8 Nxa5 Rxa5 Rfd1 Qa8 Bxb5 Rxa3 Rxa3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 Nbd7 cxd5 Nxd5 Bxd8 Nxc3 Bh4 Nd5 Bf2 c5 Bb5 Rfd8 e4 Ne7 Ne2 Bc6 Ba6 b5 a4 bxa4 dxc5 Ne5 Nd4 Rab8 Bg3 f6 O-O-O Kf7 f4 Ng4 f5 e5 Bc4 Ke8 Nxc6 Nxc6 Rxd8 Kxd8 Re1 h5 h3 Nh6 Rd1 Kc7 Rd6 Na5 Bd5 Nb3 Kd1 Nd4 Ra6 Rxb2 Rxa7 Kb8 Rxa4 Rxg2 Rxd4 Rxg3 Rb4 Kc8 c6 Nxf5 Be6 Kc7 exf5 Kxc6 h4 Kc5 Rb7 Rg4 Ke2 Kc6 Ra7 Kb6 Rd7 Kc5 Ke3 Kc6 Rf7 e4 Kd4 Kb6 Bd5 Rxh4 Rxg7 Rh2 Bxe4 Rd2 Ke3 Rd6 Rh7 Kc5 Rxh5 Ra6 Rh2 Ra3 Kf4 Ra4 Rc2 Kd6 Rd2 Ke7 Rd5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 Bg5 Nbd7 cxd5 exd5 e3 Be7 Bd3 O-O Qc2 Re8 O-O Nf8 Rab1 a5 a3 Bd6 Rbe1 Bg4 Nd2 Bh5 f4 h6 Bh4 Bg6 f5 Bh7 h3 N8d7 Nf3 b5 e4 b4 e5 bxc3 exd6 Rxe1 Rxe1 cxb2 Qxb2 Rb8 Qc3 Qb6 Bxf6 Nxf6 Rb1 Qxb1 Bxb1 Rxb1 Kh2 Bxf5 Ne5 Kh7 Nxf7 Be4 Ne5 Rb7 Kg1 a4 Nxc6 Rb3 Qd2 Rxa3 Ne5 Ra1 Kh2 a3 d7 Nxd7 Nxd7 Rb1 Nc5 Rb2 Nxe4 Rxd2 Nxd2 a2 Nb3 Kg6 Kg3 Kf5 Kf3 h5 g4 hxg4 hxg4 Kg5 Kg3 g6 Na1 Kf6 Kf4 Kf7 Ke5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 Qxd5 Nf3 Qf5 Qxf5 exf5 a3 Bd6 Bg5 Nbd7 e3 c6 Bd3 h6 Bh4 g6 Nd2 Be7 f3 Nd5 Nxd5 Bxh4 g3 Bxg3 hxg3 cxd5 Rc1 Kd8 e4 fxe4 fxe4 dxe4 Nxe4 f5 Nd6 Ke7 Nb5 a6 Nc7 Rb8 Nd5 Kd6 Nf4 g5 Ng6 Re8 Kd2 Nf6 Rxh6 Ng4 Rh5 Rg8 Rxg5 Be6 Re1 Nh6 Nf4 Bd7 Rh5 Ng4 Bxf5 Nf6 Rh6 Rbf8 Bxd7 Kxd7 Ng6 Re8 Ne5 Ke6 Ng4 Kf5 Nxf6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 c5 Nc3 d5 cxd5 Nxd5 d4 Nxc3 bxc3 g6 e4 Bg7 Bb5 Nc6 O-O O-O Be3 cxd4 cxd4 Bg4 Bxc6 bxc6 Rc1 Bxf3 Qxf3 Bxd4 Rfd1 c5 Rxc5 Bxe3 Rxd8 Bxc5 Rxa8 Rxa8 Qc3 Bb6 Kf1 Rd8 a4 e6 a5 Bd4 Qc7 Rf8 e5 h5 Ke2 Kg7 f4 Rh8 Kf3 Bg1 Qc1 Bd4 Ke4 Rd8 Qc7 Rd5 g3 Bg1 h3 Rd1 g4 hxg4 hxg4 Rd4 Kf3 Rd5 Ke2 Bd4 Kf3 g5 f5 exf5 e6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 Nbd7 cxd5 Nxd5 Bxd8 Nxc3 Bh4 Nd5 Bf2 f5 Bb5 c6 Bd3 c5 Ne2 Rac8 O-O cxd4 Nxd4 Ne5 Be2 Nc4 Rfc1 Ncxe3 Nxe6 Rfe8 Rxc8 Bxc8 Nd4 Bd7 Bxe3 Nxe3 Rc1 Rd8 Kf2 f4 g3 Nf5 Nxf5 Bxf5 Ke1 fxg3 hxg3 Rd7 b4 Kf7 Rc4 g5 f4 Be6 Rc6 Re7 Kf2 gxf4 gxf4 Kg7 Bd3 Bd5 Rd6 Bb3 f5 Rf7 Rg6 Kf8 Rxh6 Ke7 Ke3 Rf6 Rh7 Rf7 Rh4 Kd6 Kd4 Rf6 Rg4 Bf7 Rg7 a5 b5 a4 Be4 Be8 Rb7 Rf8 Rxb6 Kc7 Re6 Bxb5 Ke5 Bd7 Re7 Kd8 f6 Re8 Kd6 Bb5 Bf5 Rf8 f7 Rh8 Be6 Ba6 Ra7 Bc8 Bxc8
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 Bg5 Nbd7 cxd5 exd5 e3 Be7 Bd3 Ne4 Bxe7 Qxe7 O-O O-O Qc2 Ndf6 Rae1 Bf5 Ne5 Nxc3 Qxc3 Bxd3 Nxd3 Ne4 Qc2 Nd6 b3 Rae8 Qd2 f6 Re2 h5 f3 Qf7 Rfe1 Rd8 Nf2 Rfe8 Qb2 Re7 e4 dxe4 fxe4 Qg6 Qd2 Nb5 d5 Red7 Qb4 a5 Qc5 Qg5 a4 Nc7 Qxa5 cxd5 e5 d4 Nd3 Nd5 exf6 Qxf6 Rf2 Qd6 Qc5 Qxc5 Nxc5 Rc7 Ne6 Re7 Nxd8 Rxe1 Rf1 Re2 Rf2 Re1 Rf1 Re7 Rd1 Nc3 Rd3 Kf8 Kf1 Ke8 Rxd4 Ne4 Nxb7 Rf7 Ke1 Re7 Kf1 Rf7 Ke2 Re7 Kf1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 Nf3 e6 e3 Nbd7 Qc2 Bd6 Bd3 O-O O-O dxc4 Bxc4 b5 Bd3 Bb7 a3 a6 b4 a5 Rb1 axb4 axb4 Qe7 e4 e5 dxe5 Nxe5 Nxe5 Bxe5 Ne2 Qe6 f4 Bc7 e5 Ra2 Bb2 Nd5 Bxh7 Kh8 Bf5 Bb6 Kh1 Qh6 Be4 Ne3 Qb3 Nxf1 Rxf1 Ra7 Rf3 Bc8 f5 Qd2 Ng3 Bf2 Nf1 Qe1 Bd3 Rd7 e6 fxe6 fxe6 Rxf3 exd7 Bxd7 gxf3 Be6 Qc3
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 c6 Bg5 Nbd7 cxd5 exd5 e3 Be7 Qc2 Nh5 h4 Ndf6 Bd3 h6 Bxf6 Nxf6 O-O-O Ng4 Kb1 Be6 g3 O-O Rhf1 Rc8 Ne2 c5 dxc5 Rxc5 Qd2 Qb6 Ned4 Bf6 Nb3 Rc7 Qa5 Qxa5 Nxa5 Ne5 Nxe5 Bxe5 Rc1 Rfc8 Rxc7 Rxc7 Rc1 Bc8 Nb3 g6 Rxc7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 e6 Nbd2 Nf6 e3 Nbd7 Bd3 Bd6 e4 e5 cxd5 cxd5 exd5 O-O O-O exd4 Ne4 Nxe4 Bxe4 Nf6 Qxd4 Nxe4 Qxe4 Re8 Qd3 Qf6 Be3 Bf5 Qd2 Be4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 Bb3 a6 Nbd2 O-O h3 Ba7 Nf1 Ne7 Ng3 Ng6 O-O h6 d4 exd4 cxd4 d5 e5 Ne4 Be3 Be6 Bc2 Nxg3 fxg3 Qe7 Qd3 Rfc8 Nh4 Nf8 Nf5 Bxf5 Qxf5 c5 Bd3 Rc6 Rae1 cxd4 Bd2 Re8 Kh1 Re6 Qf3 Bb8 Qxd5 Bxe5 Rf3 Bf4 Rxf4 Rxe1 Bxe1 Qxe1 Kh2 Re7 Rxd4 Rd7 Qxd7 Nxd7 Rxd7 g5 Rxb7 h5 Bxa6 h4 gxh4 Qe5 Kh1 Qe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 h6 O-O Bc5 c3 d6 Re1 Bg4 Nbd2 Qd7 b4 Bb6 a4 a6 Nf1 g5 Be3 Ba7 Bxa7 Nxa7 Ne3 Be6 Nd2 Ke7 d4 Rag8 Bf1 c6 Qc2 g4 Rad1 exd4 cxd4 Qc7 d5 Bd7 dxc6 Bxc6 Nd5 Nxd5 exd5 Kd7 Qf5 Kd8 Qf6 Kc8 dxc6 Nxc6 b5 axb5 axb5 Nd8 Rc1 Nc6 bxc6 bxc6 Ne4 Rd8 Ba6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d6 Nf3 O-O O-O Re8 Nc3 c6 e4 Bg4 h3 Bxf3 Bxf3 e5 d5 c5 Be3 a6 a3 Nbd7 Be2 Nf8 b4 b6 Rb1 Rb8 Qa4 Qd7 Qxd7 N6xd7 Rfc1 Kh8 Na4 Bf6 bxc5 bxc5 Bd2 Rxb1 Rxb1 Rb8 Rxb8 Nxb8 Ba5 Nbd7 Bc7 Be7 Nc3 Nf6 Bd1 h5 Ba4 N8d7 Bc6 Kg7 Kg2 g5 f3 Kg6 g4 h4 Nd1 Kg7 Ne3 Kf8 Nf5 Ke8 Kf1 Kf8 Ke2 Ne8 Ba5 Nb8 Ba4 Nf6 Bc7 Nfd7 Kd3 f6 Kc2 Kf7 Kb3 Kf8 Bc6 Kf7 Ka4 Nxc6 dxc6 Nf8 Bb6 Ke8 c7 Kd7 Nxe7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 Ne5 c6 c4 Ne4 cxd5 cxd5 Nd2 Nxd2 Bxd2 Nc6 Nxc6 bxc6 Bb4 Re8 Qd2 e5 dxe5 Bxe5 Rac1 Bd7 Bc3 Bg7 Bxg7 Kxg7 e3 Rb8 Rc5 Qb6 Qc3 Kg8 b3 Qb4 Qxb4 Rxb4 Rfc1 Kg7 Ra5 Rb7 Bf1 Kf6 Ba6 Rb6 Bd3 Rb7 Kf1 Bf5 Rxc6 Re6 Rxe6 fxe6 Ke2 Bxd3 Kxd3 Ke7 f4 Kd6 Ra6 Kd7 g4 h6 h4 Ke7 g5 h5 Kd4 Kf7 Ke5 Re7 b4 Re8 Rxa7 Kg8 b5 Rb8 a4 Rb6 Ra6 Rb7 Rxe6 Kf7 b6 Rd7 a5 d4 exd4 Rd8 b7 Rb8 a6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 Nf6 O-O d5 Nxe5 Bd6 Nxc6 Bd7 Nxd8 Kxd8 Bxd7 Kxd7 exd5 Rhe8 c4 c6 Nc3 g6 d4 h5 Bg5 Nh7 Be3 a6 Qb3 Rab8 c5 Bc7 d6 Ba5 Qxf7 Kd8 Rfe1
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 h6 Bh4 b6 Nf3 Bb7 O-O Nbd7 Qc2 c5 Rad1 c4 Bf5 a6 Ne5 b5 a3 Nb6 f4 Qd6 Qf2 Bc8 Bb1 Na4 Bxf6 Bxf6 Qc2 Rd8 Qh7 Kf8 Nxd5 Qxd5 Be4 Qe6 Bxa8 Bd7 Bf3 Nxb2 Bg4 Qd6 Bxd7 Nxd1 Nc6 g6 Nxd8 Qxd7 f5 Nxe3 Qxh6 Bg7 Qxe3 Bxd4 Ne6 fxe6 fxe6 Kg8 exd7 Bxe3 Kh1 Bg5 Rd1 Bd8 Re1
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d3 d5 Nd2 Nf6 Ngf3 Bc5 e5 Nfd7 c3 O-O d4 Bb6 Bd3 c5 Nf1 f5 Ng3 Nc6 Ne2 cxd4 cxd4 Re8 h4 Nf8 h5 Nb4 Bb1 Bd7 a3 Na6 h6 g6 Bg5 Qc7 Bd3 Bc6 g4 Qd7 Rg1 Bb5 Nf4 Bxd3 Qxd3 Bd8 Nh5 fxg4 Bxd8 Qxd8 Nf6 Kh8 Ng5 Re7 Rxg4 Qa5 Kf1 Nc7 Rf4 Qa6 Qxa6 bxa6 Rc1 Rb8 Rc6 Rxb2 Nxd5 Rb1 Kg2 Re8 Rxc7 exd5 Rg7 Rb7 Rff7 Ne6 Rxh7 Kg8 Rxb7 Nf4 Kh1 Nh5 Rhg7 Nxg7 Rxg7 Kf8 e6 Rxe6 Nxe6 Ke8 h7 g5 h8=R
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 g6 Bxf6 exf6 e3 f5 c4 Bg7 Nc3 d6 g3 Nd7 Bg2 O-O Nge2 Nf6 O-O c6 b4 Rb8 a4 a6 Qd3 d5 cxd5 Nxd5 Nxd5 cxd5 Nf4 Be6 Qb3 b5 Bxd5 Bxd5 Qxd5 bxa4 Rxa4 Qb6 Rfa1 Rfd8 Qa2 Bf8 h4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 b3 Nf6 e5 Nd5 Bb2 Be7 g3 O-O Bg2 f5 O-O Nc6 c4 Nc7 Re1 b6 d4 cxd4 Nxd4 Bb7 Nc3 Rc8 Ndb5 Nxb5 Nxb5 a6 Nd6 Bxd6 Qxd6 Qc7 Rad1 Rf7 f4 Nd8 Qxc7 Rxc7 Bd4 Bxg2 Bxb6 Rc8 Kxg2 Nb7 Re2 Kf8 Red2 Ke8 Kf3 Rc6 Be3 Nc5 Bxc5 Rxc5 Rd6 Ra5 R1d2 g5 Rb6 gxf4 gxf4 Rg7 Rb8 Ke7 a4 h5 Rxd7 Kxd7 Rb7 Kc6 Rxg7 Kc5 Rb7 Kd4 Rb6 Kc3 Ke3
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 d6 Ng5 Be6 Nxe6 fxe6 Bxe6 Qe7 Bh3 a6 O-O Qf7 Nc3 Ne7 f4 g6 fxe5 dxe5 Bg5 Bg7 d4 exd4 e5 dxc3 exf6 cxb2 Qd7 Kf8 fxe7 Kg8 Be6 bxa1=Q Bxf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Be2 g6 O-O Bg7 Bb5 Bd7 Bxd7 Nbxd7 Re1 O-O d4 cxd4 cxd4 e6 Nc3 a6 Bf4 Qe7 a4 Rac8 Qd2 Rfd8 Bg5 e5 a5 Qf8 Qe2 Re8 d5 Bh6 Bxh6 Qxh6 Qd2 Qxd2 Nxd2 h6 f3 Nh5 g3 g5 Rec1 Nc5 Nc4 Red8 Nxd6 Nb3 Nxc8 Rxc8 Ne2 Nxc1 Rxc1 Rxc1 Nxc1 Ng7 Nd3 Ne8 Nxe5 Nd6 Kf2 f6 Ng4 Kg7 Ne3 Kf7 Ke2 Ke7 Kd3 Kd7 Nf5 Nf7 f4 Kc7 Kd4 h5 h4 gxh4 gxh4 Kd7 e5 fxe5 fxe5 Nd8 Kc5 Ke8 e6 Nxe6 dxe6 b6 Kd6 bxa5 e7 Kf7 Kd7 Kf6 Nd6 a4 e8=Q a3 Qe6 Kg7 Nf5 Kh7 Qf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 Bb3 Bg4 h3 Bh5 Nbd2 Qd7 Qe2 O-O-O Nf1 d5 Ng3 Bxf3 Qxf3 d4 Bc2 Ne8 O-O Nd6 Bd2 Rdf8 b4 Bb6 c4 f5 exf5 Nxf5 Nxf5 Rxf5 Qe4 a5 c5 Ba7 a3 Ne7 Bb3 c6 Bc4 a4 Rab1 Rhf8 f4 Bb8 b5 Nd5 bxc6 bxc6 Ba6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Be6 cxd5 cxd5 Nf3 Nc6 Bf4 g6 e3 Nh6 Bb5 Nf5 Ne5 Bh6 Nxc6 bxc6 Bxc6 Bd7 Bxa8 Qxa8 Bxh6 Nxh6 O-O e6 Rc1 Nf5 Qd3 Qb7 b3 Bc6 Rc2 a6 Rfc1 O-O Ne2 Bb5 Qd2 Re8 Nf4 e5 dxe5 g5 Nh5 Ne7 Nf6 Kg7 Nxe8 Bxe8 Rc7 Qb5 Rxe7 Qf1 Rxf1
Result: 1-0
Processing game...
FEN: 
Moves: e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 f4 c5 c3 Nc6 Ndf3 Qb6 a3 c4 h4 Be7 h5 h6 Ne2 Na5 g4 Nf8 Bh3 Nb3 Rb1 Nxc1 Qxc1 Bd7 Qc2 O-O-O Ng3 Kb8 Ke2 Rc8 Kf2 Be8 Kg2 Bd7 Rbf1 Qd8 Rf2 Rc7 Rhf1 Qe8 g5 Bc8 Bg4 Bd8 Qd2 Nh7 g6 Nf8 gxf7 Rxf7 Nh2 Nh7 Qc2 Bd7 Qb1 Bc8 Bd1 Bh4 Bc2 Nf8 Kh3 Bxg3 Kxg3 Rc7 Bg6 Nxg6 Qxg6 Qxg6 hxg6 Rf8 Nf3 Re7 Nh4 Kc7 Kg4 a6 Kh5 Kd8 f5 exf5 Nxf5 Bxf5 Rxf5 Rxf5 Rxf5 Kd7 a4 Ke6 Rf8 Rc7 a5 Re7 Rg8 Rd7 Rxg7 Rxg7 Kxh6 Rg8 Kh7 Rc8 g7 Kf7 e6 Kxe6 g8=Q Rxg8 Kxg8 Ke7 Kg7 Ke6 Kg6 Ke7 Kf5 Kd7 Ke5 Kc6 Ke6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Nbd7 e3 Be7 Bd3 O-O Nge2 c6 O-O Re8 f3 h6 Bh4 b6 Qc2 a5 e4 Ba6 Bxa6 Rxa6 e5 Nh7 Bf2 Ra8 f4 Rc8 Ng3 Ndf8 Rad1 Ne6 f5 Nc7 f6 Bf8 fxg7 Bxg7 Nf5 Rf8 Bh4 Ng5 Ne2 f6 Qxc6 Nce6 Qd6 Re8 Qxd8 Rcxd8 Nxg7 Nxg7 Rxf6 Re6 Rff1 Rc6 Bxg5 hxg5 Nc3 Rg6 Nb5
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 c5 cxd5 cxd4 Qa4 Bd7 Qxd4 exd5 Qxd5 Nc6 Nf3 Nf6 Qd1 Bc5 e3 Qe7 Be2 O-O-O O-O g5 b4 Bxb4 Qc2 g4 Nd4 Nxd4 exd4 Kb8 Bf4 Ka8 Nb5 Bxb5 Bxb5 Rxd4 Be3 Rdd8 Rac1 Bd6 Qa4 Bb8 Rfe1 Qd6 g3 h5 Rc6 Qe5 Ra6 Rd4 Qxd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 Bg5 g6 Nd2 Bg7 e4 fxe4 Nxe4 Nf6 Nxf6 exf6 Be3 O-O h4 d6 h5 Be6 hxg6 hxg6 Qd2 Kf7 Ne2 Rh8 Rxh8 Qxh8 Nf4 Nd7 O-O-O c6 Nxe6 Kxe6 d5 Kf7 dxc6 bxc6 Qxd6 Ne5 f4 Bf8 Qc7 Be7 fxe5 Qe8 Bc4 Kf8 Bh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nc6 Nge2 Be6 O-O h6 Bh4 Qd7 a3 Bf5 Bxf6 Bxf6 Bxf5 Qxf5 Qb3 Rab8 Nxd5 Qe6 Nef4 Qd6 Nxf6 Qxf6 Nd3 b6 Rac1 Rbc8 Qa4 Ne7 Qxa7 Nd5 Qa6 Rfe8 Qc4 Ne7 Rc2 Ng6 Qc6 Qf5 Rd1 Re7 Qf3 Qxf3 gxf3 Rd8 Rc6 Nh4 f4 Nf5 Rdc1 Rdd7 Ne5 Rd6 Rxc7 Rxc7 Rxc7 f6 Nd3 Kf8 b4 Ne7 b5 Nd5 Rc8 Kf7 f5 Ne7 Rc7 Rd5 Nb4 Rd8 Nc6 Re8 d5 Kf8 Nxe7 Rxe7 d6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c6 d4 d5 e5 Bf5 Nf3 e6 a3 Nd7 Nbd2 Ne7 Nh4 Bg6 c3 Nf5 Nxg6 hxg6 Nf3 Be7 Bd3 Nh4 Nxh4 Rxh4 g3 Rh8 h4 Qc7 Qf3 Rf8 Bg5 Qd8 Qe3 Bxg5 hxg5 Qe7 Rh7 Rg8 f4 O-O-O Ke2 Rh8 Rah1 Rxh7 Rxh7 Rg8 g4 Qd8 Kf1 Rh8 Qh3 Rxh7 Qxh7 Qf8 Ke2 Nb6 f5 gxf5 gxf5 Na4 f6 gxf6 exf6 Nxb2 g6 fxg6 Qg7 Qd6 Qh8 Qd8 Qg7 Qd6 Qh6 Qg3 Qf8 Kc7 Qe7 Kb6 Qd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 c6 c4 d5 Nf3 Nf6 e3 e6 Bd3 Be7 O-O O-O Nc3 Nbd7 b3 Re8 Bb2 b6 e4 dxe4 Nxe4 Qc7 Qe2 h6 Nxf6 Bxf6 Ne5 Bb7 Rad1 Rad8 Ng4 Bg5 f4 Bxf4 Qe4 f5 Qxf4 Qxf4 Rxf4 fxg4 Rxg4 g5 Rg3 Nf6 Rh3 Kg7 Rf1 g4 Re3 Bc8 d5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Be2 a6 O-O e6 Re1 Be7 Bf1 e5 h3 O-O d4 cxd4 cxd4 Qc7 Nc3 Nbd7 Be3 b5 Rc1 Qa5 Nd2 Qd8 d5 Bb7 b4 h6 Nb3 Nh7 Na5 Qb8 a4 Bc8 Nc6 Qb7 Nxe7 Kh8 Nc6 f5 axb5 axb5 Nxb5 Qa6 Nxd6 Qa4 exf5 Ndf6 Qxa4 Rxa4 Ne7 Ba6 Ng6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 g6 Bxf6 exf6 e3 Bg7 h4 d6 h5 Be6 Ne2 c6 Nf4 g5 Nxe6 fxe6 h6 Bf8 Qh5 Ke7 Nd2 Qe8 Qh3 Nd7 O-O-O Kd8 e4 Qf7 d5 e5 Nc4 Kc7 Ne3 Rb8 Bc4 c5 a4 a6 a5 b5 axb6 Rxb6 g3 Rg8 Qf1 Nb8 Nf5 Qd7 b3 a5 Kd2 a4 Ra1 axb3 cxb3 Qd8 Ra7 Rb7 Qa1 Qc8 Qa5 Kd7 Bb5
Result: 1-0
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 f3 g6 Be3 Bg7 Qd2 O-O c4 Na6 Nc3 Re8 Nge2 c6 g4 Nd7 h4 Nf8 h5 e5 d5 cxd5 cxd5 Bd7 Ng3 Nc7 Bg5 f6 Be3 b5 a3 a5 Bd3 g5 Kf2 h6 b4 a4 Rac1 Rb8 Rc2 Rb7 Rhc1 Qb8 Nd1 Rc8 Kg2 Rd8 Bf2 Na6 Ne3 Rc7 Nef5 Rxc2 Rxc2 Kf7 Qe2 Be8 Qe3 Nd7 Rc6 Nc7 Nxg7 Kxg7 Nf5 Kh7 Rxd6 Nf8 Rxf6 Bd7 Rxh6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 e6 e4 Be7 Nd2 O-O Ngf3 d5 e5 Nfd7 h4 c5 Bd3 h6 c3 f5 exf6 Bxf6 Qe2 Qe8 Bxf6 Rxf6 g4 Rf8 g5 Qh5 Qxe6 Qf7 Qxf7 Rxf7 O-O-O Re7 Rde1 Nc6 Rxe7 Nxe7 Re1 Kf8 Nf1 cxd4 cxd4 Nb6 Ng3 Bg4 Ne5 Rc8 Kd2 Bh3 Nh5 Nc4 Bxc4 dxc4 Nf4 Bf5 Kc3 a5 a4 b6 d5 Rd8 Rd1 Bd7 d6 Nf5 Nfg6 Kg8 Ne7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 O-O O-O a6 Re1 b5 Bb3 d6 h3 Re8 Nbd2 h6 Nf1 Bb7 Ng3 Na5 Bc2 c6 b4 Bb6 bxa5 Bxa5 Bd2 Bb6 d4 c5 d5 c4 Nf5 Nh7 Nh2 Bc8 Ng4 f6 Be3 Bxe3 Rxe3 h5 Nh2 g6 Rg3 Nf8 Qxh5 Bxf5 exf5 Kf7 Rxg6 Nxg6 Qh7 Kf8 fxg6 Ra7 Qh8 Ke7
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 d6 d4 Bd7 Nc3 Nf6 Bxc6 Bxc6 Qd3 d5 Nxe5 dxe4 Qc4 Bd5 Qb5 c6 Qxb7 Rc8 Nxd5 Be7 Nxf6 gxf6 Nxc6 Qc7 Qxc7 Rxc7 Nxe7 Kxe7 c3 Rg8 g3 f5 Bf4 Rb7 O-O-O Rc8 d5 Kd7 Rd4 Rc5 Rhd1 Rcb5 b4 a5 a4 R5b6 b5 Ra7 c4 Ke8 c5 Rbb7 d6 Rd7 c6 Rd8 b6 Ra6 d7 Ke7 c7 Rxb6 cxd8=Q Kxd8 Bc7 Kxc7 d8=Q Kb7 Rd7 Ka6 Qa8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Bb5 e6 c3 Nge7 d4 cxd4 cxd4 d5 Nc3 a6 Ba4 g6 Bg5 Bg7 Bxc6 bxc6 e5 O-O Qd2 Qc7 Na4 a5 O-O Ba6 Rfc1 Rfc8 Nc5 Nf5 g4 Ne7 Bf6 Kf8 Ng5 Kg8 Bxg7 Rab8 Bf6 Qd8 Nxh7 Kxh7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nc6 d4 cxd4 cxd4 e6 Nc3 Bd7 Bd3 Qc7 O-O a6 d5 Nce7 Be3 Nf6 Rc1 Qd8 Qe2 Rc8 dxe6 Bxe6 Nd5 Qd7 Rxc8 Qxc8 Nxf6 gxf6 Nd4 Bd7 Qf3 Bg7 Rc1 Qd8 Nf5 O-O Nxg7 Kxg7 Qh5 Rg8 Qh6 Kh8 Qxf6 Rg7 Bh6 Qg8 Bxg7 Qxg7 Qxe7 Be6 Qd8 Qg8 Qxd6 Bh3 Qf6 Qg7 Qxg7 Kxg7 gxh3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 Bg5 Nc2 O-O a4 bxa4 Rxa4 a5 Bc4 Rb8 b3 Kh8 O-O Bd7 Qd3 f5 Ra2 g6 f3 Bh6 Rd1 Qh4 Nde3 Rf6 Bd5 Rbf8 Nf1 Ne7 Nce3 Nxd5 Nxd5 fxe4 Qxe4 Qxe4 fxe4 R6f7 Rxa5 Bg4 Rda1 Be6 Nfe3 Bxe3 Nxe3 Bxb3 c4 Rc8 Rc1 Rb7 Ra6 Rb4 Rxd6 Ba4 Re6 Rb3 Nd5 Bb5 Re7 Ba6 c5 Rf8 c6 Rb2 h4 Bd3 Rxe5
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 c6 Nc3 d5 cxd5 cxd5 Ne5 e6 O-O Nfd7 f4 Nc6 Be3 f6 Nf3 f5 Ne5 Nb6 b3 Bd7 Qd3 Nc8 Bc1 Nxe5 dxe5 Rf7 Be3 Bc6 Rfc1 Qa5 a3 a6 Bd4 Bf8 e3 Be8 Qd2 Qd8 Bf1 Rc7 Qb2 Rc6 Na4 Rxc1 Rxc1 Bc6 Nc5 Qe8 a4 a5 Qc3 Qf7 Bb5 Bxb5 axb5 h6 Na4 Ba3 Qxc8 Rxc8 Rxc8 Kh7 b6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 Nbd7 g4 O-O O-O-O Qc7 Kb1 Rfc8 g5 Nh5 Rg1 Nb6 Qf2 Bd8 h4 g6 Rd2 Qc6 f4 exf4 Bxf4 Nxf4 Qxf4 Qe8 Nd4 Rxc3 bxc3 Na4 Rg3 Ba5 Nxe6 Qxe6 Rd4 Nxc3 Rxc3 Bxc3 Bc4 Qe7 Qxd6 Re8 Qxe7 Rxe7 Rd8 Kg7 Bd5 Be1 h5 gxh5 Rb8 b5 Rb7 Rxb7 Bxb7 a5 Kc1 h4 Kd1 Bg3 Ke2 h3 Kf3 h2 Kg2 b4 Bd5 Be5 Bb3 Kg6 Kh1 h6 gxh6 f6 Kg2 Kxh6 Ba4 Kg5 Bc6 Kg4 Ba4 Bg3 Bb5 Kf4 Bc6 Ke3 Kh1 Be5 Kg2 Kd2 Ba4 Kc3 Kh1 Kb2 Bb3 Ka3 Bd5 Kb2 Bb3 Kc3 Ba4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 d4 g6 c4 Bg7 g3 c5 Bg2 Qa5 Nc3 Ne4 Qd3 cxd4 Nxd4 Nc5 Qd1 Nc6 e3 Ne6 O-O Qc5 Ndb5 a6 Na4 Qxc4 Nbc3 Rb8 Nb6 Qc5 Ncd5 O-O b4 Qd6 Rb1 Nc7 Nc4 Qe6 e4 d6 Be3 Qd7 Ncb6 Qd8 f4 Nxd5 exd5 Na7 Rc1 Bf5 g4 Nc8 Nc4 Bd7 Qe2 Bb5 Qf2 e6 Rfd1 e5 f5 gxf5 gxf5 Qf6 a4 Bxc4 Rxc4 Ne7 Rf1 Kh8 Qg3 b5 Rh4 Nxf5 Rxh7 Kxh7 Be4 Kg8 Bxf5 e4 Qh3
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 Qd2 Be7 g4 h6 O-O-O b5 Kb1 Nb6 Na5 Qc7 Bxb6 Qxb6 Nd5 Nxd5 exd5 Bd7 h4 b4 Nc6 Bxc6 dxc6 Qxc6 Qxb4 d5 Qb3 Rd8 c4 O-O Rxd5 Rxd5 cxd5 Qd6 Qc4 e4 Qxe4 Rb8 Bd3 g6 h5 Qf6 Rh2 Ba3 b3 Rb7 Qe3 Re7 Be4 Rc7 Rc2 g5 Qd2 Rxc2 Kxc2 Qa1 Kd3 Qf1 Kd4 Qa1 Kc4 Qf1 Qd3 Qf2 b4 Qxa2 Qb3 Qxb3 Kxb3 Bc1 Bd3 Bf4 Bxa6 Kf8 Kc4 Ke7 Bc8 Bd6 Bf5 Kf8 Bh7 Ke7 b5 Kd7 Bg8 Ke7 b6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 e6 c4 c5 g3 cxd4 Nxd4 Qb6 Bg2 Bc5 e3 Nc6 Nf3 O-O O-O Rd8 Nc3 Qa6 e4 Qxc4 Bg5 Be7 e5 Ne8 Bxe7 Nxe7 Ng5 h6 Nge4 Nc6 Nd6 Qc5 Re1 Nxd6 exd6 b6 Rc1 Qa5 b4 Qxb4 Re4 Qa5 Ra4 Qe5 Re4 Qa5 Ra4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 d6 Be3 Nf6 Bc4 Be7 Qe2 a6 Bb3 Qc7 O-O-O O-O Rhg1 Nd7 Kb1 Nc5 Qh5 Nxb3 axb3 Qa5 g4 Qxh5 gxh5 Re8 Bh6 g6 Na4 Nxd4 Rxd4 b5 Nb6 Rb8 Nxc8 Rexc8 Rd3 Rc5 hxg6 hxg6 Bf4 Rh5 Bxd6 Bxd6 Rxd6 Rxh2 Rf1 Ra8 Rb6 Kg7 c4 bxc4 bxc4 a5 Ka2 Rh3 f3 g5 Rb5 Kf6 b3 a4 b4 Rh2 Ka3 Rc2 c5 Rc3 Kb2 Re3 Rf2 Ke5 c6 Kd6 Rc5 Rb3 Ka2 Rxb4 Rc3 Kc7 Rd2 Rd8 Rxd8 Kxd8 Rd3 Kc7 Rd7 Kxc6 Rxf7 g4 fxg4 Rxe4 Rg7 Re3 Rg8 Rg3 g5 Kd5 g6 Ke5 g7 Kf6 Ra8 Rxg7 Rxa4 Rb7 Rf4 Kg5 Rf1 e5 Rg1 Kf5 Rf1 Ke4 Re1 Kd4 Rd1 Kc3 Re1 Re7 Rh1 e4 Kb1 e3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 Nc3 Bg7 g3 d6 Bg2 O-O O-O e5 d4 Nbd7 e4 a6 h3 exd4 Nxd4 Re8 Re1 Rb8 Bg5 h6 Be3 Ne5 b3 c5 Nf3 Nxf3 Qxf3 b5 Rad1 bxc4 Bf4 Re6 bxc4 Rb4 e5 Qe8 Nd5 Nxd5 cxd5 Rxe5 Bxe5 Bxe5 Qf6 Bd7 Qf3 h5 Qa3 Bb5 Kh1 Qd7 Rxe5 dxe5 Qe3 Qd6 Re1 Rb2 Qxe5 Qxe5 Rxe5 Rxa2 d6 Rd2 Rxc5 Rxd6 g4 hxg4 hxg4 Rd4 g5 Rc4 Re5 Rf4 Kg1 Kf8 Be4 Bd7 f3 Bf5 Bb7 Ra4 Kf2 Be6 Ke3 Ra1 Kf4 a5 Bc6 Ke7 Rb5 Kd6 Be8 a4 Ra5 a3 Ra6 Ke7
Result: 0-1
Processing game...
FEN: 
Moves: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 Bc4 e6 N1e2 Bd6 Nf4 Qh4 Ngh5 Bxh5 Qxh5 Qxh5 Nxh5 g6 Nf4 Nf6 Nd3 Nbd7 O-O a5 a4 Nd5 c3 O-O Bh6 Rfe8 Rfe1 Nc7 g3 b5 Bb3 Nb6 Ne5 Bxe5 dxe5 Ncd5 h4 Reb8 Re4 Nxa4 Bxa4 bxa4 Rexa4 Rxb2 Rxa5 Rxa5 Rxa5 Rb8 c4 Nb6 Rc5 Rc8 g4 Nd7 Ra5 Nb6 Rc5 Nd7 Ra5 Nb6 Rc5 Nd7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 Bg5 Nc2 O-O a4 bxa4 Rxa4 a5 Bc4 Rb8 b3 Kh8 O-O g6 Qd3 Bd7 Ra2 f5 f3 Bh6 Rd1 Qh4 Nce3 Be6 Nf1 fxe4 Qxe4 Qxe4 fxe4 Rf7 Rf2 Kg7 Rxf7 Bxf7 Kf2 Nd8 Ra1 Ne6 Rxa5 Nc5 Ng3 Nxb3 Ra7 Nd2 Ba2 Rb2 Ke1 Kf8 Nb4 Nb3 Nd3 Rxg2 Rxf7 Kxf7 Bxb3 Kg7 Nf1 Bf4 h3 Bg3 Kd1 h5 Be6 Bf4 Ne1 Rf2 Bc4 g5 Be2 g4 Nd3 Rg2 hxg4 h4 Bf3 h3 Ke1 Rc2 Nxf4 exf4 Nd2 h2 Nb1
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6 O-O Be7 a4 Nc6 Be3 O-O f4 Qc7 Kh1 Re8 Bf3 Rb8 g4 Nd7 Bg2 b6 g5 Bb7 Qg4 Bf8 f5 exf5 Qxf5 Nde5 Nd5 Qd8 Nb3 Ne7 Qf4 Rc8 Nd4 Nxd5 exd5 Qd7 b3 g6 c4 Re7 Bg1 Rce8 Rae1 Bg7 Qg3 Qc7 Rd1 Nd7 Rde1 Nc5 Nc6 Rxe1 Rxe1 Rxe1 Qxe1 Nxb3 Qe8 Bf8 Bh3 Na5 Qd8 Qxd8 Nxd8 Ba8 Bf1 Nb7 Nxb7 Bxb7 Bxb6 h6 h4 hxg5 hxg5 Be7 Bc7 Kf8 Kh2 Ke8 Bh3 Bxg5 Bxd6 Be7 Be5 Bc5 Bf6 a5 Kg3 Ba6 Bf1 Bc8 Bd3 Kd7 Kf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 c6 Bg2 Bg7 Nf3 d5 cxd5 cxd5 O-O O-O Nc3 Ne4 Nd2 f5 Ndxe4 dxe4 Qb3 Kh8 Rd1 Nc6 Be3 Bd7 Rac1 Rc8 Qa3 Qa5 Qxa5 Nxa5 b3 Nc6 f3 exf3 exf3 f4 Bf2 e5 Ne2 Nxd4 Nxd4 exd4 Rxc8 Rxc8 Bxd4 Bxd4 Rxd4 Rc1 Kf2 Rc2 Ke1 Bc6 Rd2 Rxd2 Kxd2 g5 gxf4 gxf4 Kc3 Kg7 Kd4 Kf6 b4 b6 a3 Bb7 Bh1 Ba8 Bg2 Bc6 Bh1 Ke6 Bg2 Kd6 Bh1 a5 bxa5 bxa5 Bg2 a4 Bh1 Ba8 Bg2 Bb7 Bh1 Ke6 Bg2 Kf6 Bh1 Kg5 Ke5 h5 Bg2 h4 h3 Ba6 Ke4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nf6 Nc3 d5 cxd5 Nxd5 d4 Nxc3 bxc3 g6 e4 Bg7 Rb1 O-O Be2 cxd4 cxd4 Qa5 Bd2 Qxa2 O-O Bg4 Bg5 h6 Be3 Nc6 d5 Ne5 Rxb7 e6 Re1 Bxf3 gxf3 Rfd8 d6 Qa5 Rf1 Bf8 d7 Qa2 Re1 Qa5 Rf1 Qa2 Bb5 a6 Bd4 Bg7 Bxe5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 d5 Nb8 Nf1 Nbd7 N3h2 Nc5 Bc2 c6 b4 Ncd7 dxc6 Bxc6 Ng4 Nxg4 hxg4 Nb6 Ne3 d5 exd5 Nxd5 Nxd5 Qxd5 Qxd5 Bxd5 Bb3 Be6 Bxe6 Rxe6 a4 bxa4 Rxa4 Rc8 Bd2 Rd6 Ra2 f6 Re4 g6 c4 f5 gxf5 gxf5 Rxe5 Rxc4 Rxf5 Bh6 Bxh6 Rxh6 Rc5 Rxb4 Rc8 Kg7 Rc7 Kg8 g3 Rbb6 Ra7 Rhg6 Kg2 Rbf6 Ra3 Kh8 f3 Kg8 Kf2 Rb6 f4 h5 Kf3 Rbc6 Rd7 Rg7 Rxg7 Kxg7 Kg2 Kh6 Kh3 Rg6 Ra5 Kg7 Rc5 Kh6 Rc8 Kh7 Ra8 Kg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bb5 c6 Ba4 O-O Ne2 c5 O-O Nc6 Be3 Na5 dxc5 Nc4 Qxd8 Rxd8 Bg5 Bd7 Bb3 Na5 Bxe7 Re8 Bd6 Rxe4 Nd4 Nxb3 axb3 Bxd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d6 Nf3 g6 c4 Bg7 Nc3 e5 dxe5 dxe5 Qxd8 Kxd8 Bd2 f6 O-O-O Bd7 g4 Kc8 h3 a5 Bg2 Ne7 Be3 Na6 Nd2 Bc6 Nd5 Nxd5 cxd5 Bb5 Ne4 Bxe2 Rd2 Bb5 d6 Bc6 Re1 Rd8 f4 Bxe4 Bxe4 c6 fxe5 Kd7 exf6 Bxf6 Rf2 Rf8 Ref1 Bg7 Rf7 Ke6 Bd3 Rxf7 Bc4 Kxd6 Rxf7 b5 Rxg7 bxc4 Rxh7 Rg8 Kd2 Nb4 a3 Nd5 Bd4 c5 Bc3 Rf8 Rh6 Nf4 h4 Kd5 Bxa5 Re8 Bc7 Nd3 Rxg6 Nxb2 h5 Na4 Rd6 Ke4 h6 c3 Kc2 Rf8 h7 Rf2 Kc1 Rf1 Rd1 Rf8 Ba5
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 h6 Bh4 O-O e3 b6 Be2 Nbd7 cxd5 exd5 O-O Bb7 Rc1 c5 Qa4 a6 dxc5 bxc5 Rfd1 Qb6 Qb3 Qa7 Bg3 Rad8 Rd2 Rfe8 Qd1 Bf8 Bh4 Qa8 Ne1 Be7 Bg3 Nf8 Nf3 Ne6 Bh4 d4 exd4 cxd4 Na4 Nf4 Nc5 Bxc5 Bxf6 d3 Bxd3 Bxf3 gxf3 Rd5 Bh4 Bb4 Rc3 Bxc3 bxc3 Red8
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Nf5 Bxf5 exf5 Qa5 O-O Bxc3 bxc3 Qxf5 Rb1 Nd7 Rxb7 Ngf6 Re1 Kf8 h3 Re8 Bxa6 h4 Bd3 Qd5 Rb5 Nc5 Bh2 Rg8 Rb4 Nxd3 cxd3 Qxa2 Rb5 Rg6 Rf5 g4 hxg4 Nxg4 Qf3 f6 Rh5 Nxh2 Kxh2 Qd2 Re4 Qxc3 Rexh4 Rg8 Rg4 Qb3 Rxg8 Qxg8 Qh3 Qg6 Rh6 Qg5 Rh8 Kf7 Qh7 Ke6 Rxe8
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Be2 b5 O-O b4 Na4 Bb7 c3 Nf6 cxb4 Bxb4 Be3 O-O Rc1 Nc6 Qd3 Qa5 Nc5 Ne5 Qb1 Bxc5 Nb3 Qc7 Rxc5 Qb8 f3 d5 Na5 Rc8 b4 Nc6 Nxb7 Qxb7 Rfc1 dxe4 fxe4 Nd7 R5c3 Nde5 a4 Rd8 b5 axb5 Qxb5 Qxb5 axb5 Nb4 Rb1 Ra2 Bf1 Rc2 Rxc2 Nxc2 Bc5 Nd4 b6 Ndc6 b7 h5 Bb6 Rd2 Ba5 Rd4 Bc3
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 Qc2 Na6 e3 Nb4 Qd2 Bf5 Rc1 a5 a3 Na6 Nge2 h6 Bf4 Nd7 Ng3 Be6 e4 Nb6 exd5 Nxd5 Nxd5 Qxd5 Bc4 Qxg2 Qe3 O-O-O Bxe6 fxe6 Qxe6 Rd7 Qe8 Rd8 Qe6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Bg5 a4 bxa4 Rxa4 a5 Bb5 Bb7 Nce3 Bxe3 Nxe3 Ne7 O-O Rb8 Qd3 Qb6 Bc4 Bc6 Ra2 Rfd8 b3 Qc5 Rfa1 Ra8 h4 h6 h5 Bb7 Rd1 Bc6 Rda1 Bb7 Bd5 Bxd5 exd5 Rdc8 b4 Qxc3 Qxc3 Rxc3 Rxa5 Rb8 Ra7 Kf8 Rd7 Nc8 Rd8 Ke7 Rg8 Rxb4 Ra8 Kd7 Rxg7 Rc1 Kh2 Rf4 Ng4 Ne7 Ra7 Ke8 g3 Rf5 Rh7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Qxd4 a6 c4 Nc6 Qd2 g6 Nc3 Bh6 Qc2 Bg7 Be3 Nf6 Be2 O-O O-O Bg4 Rad1 Nd7 b3 Rc8 a4 Qa5 Nd4 Bxe2 Ndxe2 Nc5 f3 Rfd8 Nd5 Rd7 b4 Nxb4 Qd2 Nxd5 Qxa5 Nxe3 Rb1 Nxf1 Kxf1 Rdc7 Nf4 e6 g3 h5 Ng2 Bd4 Rd1 e5 Qb4 Rc6 a5 R8c7 Ke2 Ne6 Rc1 b5 axb6 Rxb6 Qa4 Kg7 Ne3 Bxe3 Kxe3 Nc5 Qa5 Rb3 Ke2 Rcb7 Rc2 Ne6 Rd2 Nd4 Rxd4 exd4 Qxa6 Re3 Kf1 Rb1 Kg2 Rb2 Kh3 Ree2 Kh4 Kh6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 Ne4 Bf4 d5 e3 Bf5 f3 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3 bxc3 c5 g4 Bg6 h4 h5 g5 Nfd7 cxd5 exd5 Bh3 O-O Ne2 Nc6 O-O Re8 Re1 Nb6 Ng3 Nc4 Bf1 N6a5 e4 cxd4 cxd4 dxe4 fxe4 Nb2 Qd2 Nac4 Qf2 Rc8 Rab1 Qa5 d5 Qxa3 d6 Red8 Bc1 Nd3 Bxa3 Nxf2 Bxc4 Nxe4 Nxe4 Rxc4 Rb4 Rxb4 Bxb4 Bf5 Nc3 a5 Ba3 Rc8 Nd5 Kh7 Ne7 Re8 Rc1 Be6 Rc7 b5 Bc5 b4 Bd4 Rd8 Be5 b3 Nc6 b2 Bxb2 Rxd6 Nd4 Bd5 Bc3 a4 Ra7 Bb3 Kf2 Rd5 Ke3 Re5 Kd2 Re4 Nxb3 axb3 Rxf7 Kg6 Rxg7 Kf5 Rf7 Kg4 g6 Kxh4 g7 Rg4 Rf4 Rxf4 g8=Q Rg4 Qf7 Rg2 Kd3 Rg3 Kc4 Rg4 Kxb3
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 Bg5 Be7 e3 O-O Rc1 Nbd7 cxd5 exd5 Bd3 c5 O-O Ne4 Bxe7 Qxe7 dxc5 Ndxc5 Nd4 Rac8 Nce2 g6 Bb1 Rfd8 Ba2 Qd7 h3 Ba6 b4 Na4 Rxc8 Rxc8 b5 Bxb5 Nxb5 Qxb5 Qxd5 Qxd5 Bxd5 Nec3 Nxc3 Nxc3 Bf3 Rc5 Ra1 Ra5 Kf1 Nb5 a4 Nc3 Ra3 Nxa4 Bc6 Nc5 Rxa5 bxa5 Ke2 f5 f4 Kf7 g4 Ke6 g5 a4 Kd2 Kd6 Be8 a3 Kc2 Ke7 Bc6 Nd3 Kb3 a2
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 O-O Bg2 cxd4 Nxd4 d5 cxd5 Nxd5 Bd2 Nxc3 bxc3 Be7 Rb1 e5 Rxb7 exd4 Rb3 Be6 Bxa8 Na6 Bf3 Bxb3 Qxb3 d3 Qc4 Nc5 O-O dxe2 Qxe2 Bf6 Be3 Qc7 c4 Re8 Qc2 Rb8 Rb1 Rxb1 Qxb1 Be7 Bd5 Qc8 Kg2 Bd6 h4 Qc7 h5 h6 Qb5 Nd3 Qe8 Bf8 c5 Nb4 Be4 Nxa2 c6
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 Bb7 Bg2 c5 O-O g6 d4 cxd4 Qxd4 Bg7 Nc3 d6 Rd1 Nbd7 Be3 O-O Qh4 Rc8 Bh3 Bxf3 exf3 Ne5 Kg2 Rxc4 Rd4 Rxc3 bxc3 Qa8 Kf1 Nxf3 Bg2 Nxh4 Bxa8 Nf5 Bb7 Nxd4 cxd4 Rb8 Ba6 Nd5 Rc1 h5 Ke2 e6 a3 b5 Rc8 Rxc8 Bxc8 Nxe3 fxe3 d5 Ba6 Bf8 Bxb5 Bxa3 h3 f5 Ba6 g5 Kf3 g4 hxg4 hxg4 Kf2 Kf7 Ke2 Bd6 Kf2 Bb4 Kf1 Bd2 Ke2 Bb4 Kf1 Ke7 Kf2 Ba5 Kf1 Bb6 Kf2 Kf6 Ke2 e5 dxe5 Kxe5 Kd3 Bc7 Ke2 Kf6 Kf2 Be5 Kg2 d4 exd4 Bxd4 Kf1 Kg5 Kg2 f4 gxf4 Kxf4 Kf1 Ke3
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Nf6 dxc5 b6 e4 bxc5 Nc3 Nc6 Bc4 d6 e5 dxe5 Qxd8 Kxd8 O-O-O Bd7 Bb5 exf4 Bxc6 Rc8 Bxd7 Nxd7 Nh3 Ke8 Nxf4 g6 Kb1 Bg7 Ne4 Nb6 Rhe1 Kf8 Nxc5 Bxb2 Kxb2 Rxc5 Rd8 Kg7 Re8 Rg8 Rxg8 Kxg8 Rxe7 Kh8 Kc1 Ra5 Rxf7 Rxa2 Rc7 a5 g4 Ra3 Ne6 Kg8 f4 Re3 f5 gxf5 gxf5 Re5 Rd7 h6 Rf7 Kh8 Rf6 Nc4 Kb1 a4 Rxh6 Kg8 Rg6 Kf7 Ng5 Kf8 f6 Rb5 Ka1 Ne3 Nh7 Ke8 Rg8 Kd7 Rg7 Kd6 f7 Nxc2 Ka2
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 Qd2 b5 g4 Nb6 O-O-O Be7 Qf2 Nfd7 Kb1 Qc7 h4 O-O g5 b4 Nd5 Nxd5 exd5 Bf5 Bd3 Bxd3 cxd3 a5 Rc1 Qb7 Rc6 a4 Nd2 Qb5 Qe2 b3 a3 Qxd5 Rc7 Qe6 Rhc1 d5 Bf2 Rfd8 R1c6 Qf5 Nf1 d4 Ng3 Qh3 Ne4 Bf8 Bg3 Re8 Be1 Be7 h5 Nf8 h6 Ne6 Rb7 Nf4 Qd1 Rac8 Rbc7 Rxc7 Rxc7 Qf1 hxg7 Qxd3 Qxd3 Nxd3
Result: 0-1
Processing game...
FEN: 
Moves: c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e6 Nc3 Bb4 g3 d5 cxd5 Nxd5 Bd2 O-O Bg2 Bxc3 bxc3 e5 Nb5 a6 Bc1 Be6 c4 Ne7 Nd6 Nbc6 O-O b6 Ba3 Ra7 c5 bxc5 Ne4 c4 Ng5 Qxd1 Rfxd1 Rc8 Nxe6 fxe6 Rac1 Nd4 Kf1 Nb5 Bxe7 Rxe7 a4 Nd4 e3 Nb3 Rc3 a5 Ke1 Rec7 Bf1 Kf7 Rd6 Ke7 Rb6 e4 Rb5 h6 h4 g6 Be2 Kf6 Bf1 Rc6 Be2 R6c7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 b6 Nc3 Bb7 a3 d5 Bg5 Be7 e3 O-O Rc1 Nbd7 cxd5 exd5 Bd3 c5 Bf5 h6 Bh4 Ne4 Bxe7 Qxe7 Bxd7 Nxc3 Rxc3 Qxd7 dxc5 bxc5 Rxc5 Ba6 Nd4 Rac8 Rxc8 Rxc8 f3 Qe7 Qd2 Qh4 Kd1 Qg5 Ke1 Qh4 Kd1 Qg5 g4 Qf6 h4 Rc4 Ke1 Bc8 Ke2 Qe5 Kf2 Qf6 Kg2 h5 g5 Qe7 b3 Rc7 Rc1 g6 a4 Rxc1 Qxc1 Qd7 Kg1 Qh3 Qe1 Bd7 Qf2 a6 Qh2 Qxh2 Kxh2
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Bd3 Bg4 Bc2 g6 h3 Bxf3 Qxf3 Bg7 O-O O-O Qd1 Nc6 d3 Rb8 Nd2 b5 a3 Nd7 Nf3 Nde5 Nxe5 Bxe5 Be3 Qb6 Kh1 Bg7 f4 a5 Qc1 b4 Ba4 Rfc8 Rb1 Qa6 c4 Nd4 Qd2 Rb6 f5 Qb7 Bxd4 Bxd4 Bb5 e6 f6 Kh8 Qg5 Qc7 Rbe1 Qd8 Re2 Qf8 a4 Rd8 b3 h6 Qh4 Kh7 Rf3 d5 e5 dxc4 Bxc4 Bc3 Re4 Rd4 Rff4 Rb8 Rxd4 Bxd4 Re4 Rd8 Qf4 Bc3 h4 h5 g4 hxg4 Qxg4 Qh6 Kg2 Rd4 h5 Qd2 Qe2 Qg5 Kf1 gxh5 Rxd4 Bxd4 Qe4 Kh6 Qa8 Qg1 Ke2 Qe3 Kd1 Qg1 Ke2 Qh2 Kd1 Bc3
Result: 0-1
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 b6 g3 c5 Bg2 Bb7 O-O g6 Nc3 Bg7 d4 cxd4 Qxd4 d6 Be3 Nbd7 Rac1 O-O Qh4 Rc8 b3 a6 Bh6 Rc5 Bxg7 Kxg7 Qd4 Qa8 Rfd1 b5 cxb5 axb5 b4 Rc4 Qe3 Qa3 Nd2 Rc7 Bxb7 Rxb7 Qxe7 Qxb4 Nb3 Re8 Qxd6 Qxd6 Rxd6 b4 Nd5 Nxd5 Rxd5 Rxe2 Rd2 Rxd2 Nxd2 Ne5 Rb1 Rd7 Nf1 Nc6 Ne3 Rd2 Rc1 Ne5 Rc2 Nf3 Kf1 Rxc2 Nxc2 b3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Bg5 d5 Bxf6 exf6 e3 Bd6 c4 dxc4 Bxc4 O-O Nf3 Nd7 Nc3 f5 O-O Nf6 a3 c6 Rc1 Qe7 Re1 g6 Qc2 Bd7 g3 Rac8 b4 Kg7 Rb1 Ra8 Nd2 Rfe8 Nb3 b6 Bf1 h5 h4 f4 e4 fxg3 e5 Ng4 fxg3 Bc7 Bd3 Rac8 Qd2 Be6 Qe2 Red8 Bc4 Bxc4 Qxc4 b5 Qe2 Bb6 Rbd1 a5 bxa5 Ba7 Qb2 b4 Ne4 bxa3 Qa1 Qxe5 Nbc5 Qd5 Qxa3 Re8 Qd3 Rcd8 Re2 Ne5 Qc3 Bxc5 Qxc5 Nf3 Kf2 Rxe4 Qxd5 Rxe2 Kxe2 Rxd5 Kxf3 Rxa5 Rc1 Ra6 Rc5 Kf6 d5 cxd5 Rxd5 Ra3 Kg2 Ke6 Rb5 Rd3 Ra5 f6 Rb5 Rd5 Rb6 Kf5 Rb4 Re5 Ra4 Re4 Ra6 Rd4 Kf3 Rd3 Kf2 g5 hxg5 fxg5 Ra5 Kg4 Rb5 Rf3 Kg2 Rxg3 Kh2
Result: 0-1
Processing game...
FEN: 
Moves: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Bd2 c6 Bd3 Bg4 Ne4 Bxd1 Nxf6 exf6 Bxa5 Bh5 f4 Bd6 f5 g6 Kf2 gxf5 Re1 Kf8 Bd2 Kg7 Nh3 Nd7 Bf4 Bf8 Bg3 Bg6 Nf4 Rd8 h4 h6 Rd1 Nb6 Be2 Bd6 h5 Bh7 c4 Rhe8 Rh4 Re4 b3 Rxd4 Rxd4 Bc5 Kf1 Bxd4 Be1 Be5 g3 Nd7 Bf2 Nf8 Be3 Ne6 Rh2 Bd4 Bxd4 Rxd4 Rf2 Ng5 Kg2 Ne4 Rf3 Rd2 a4 Nc5 Kf1 a5 Re3 Kf8 Ke1 Rb2 Bd1 Bg8 Re2 Rb1 Re3 Rb2 Re2 Rb1 Re3 Rb2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 c5 c4 Nf6 Nc3 d5 cxd5 Nxd5 d4 Nxc3 bxc3 g6 e4 Bg7 Be3 Qa5 Qd2 Bg4 Rc1 Bxf3 gxf3 e6 d5 exd5 exd5 Nd7 c4 Qb6 Bh3 f5 O-O Qd6 Bf4 Be5 Rfe1 O-O-O Bxe5 Nxe5 Qc3 Rhe8 Re3 Qf6 f4 Nd7 Bg2 Qxc3 Rcxc3 Nf6 Bf3 Rxe3 fxe3 Rd6 Ra3 Kb8 Rb3 Ra6 a3 Ne8 e4 fxe4 Bxe4 Nd6 Bd3 Ra4 Rc3 Kc7 Kf2 b5 cxb5 Kb6 Ke3 c4 Bc2 Nxb5 Bxa4 Nxc3 Be8 Kc5 d6 Nd5 Ke4 Nf6 Ke3 Kxd6 Bf7 Kc5 h4 Ng4 Kd2 Kd4 h5 gxh5 Bxh5 Nf6
Result: 0-1
Processing game...
FEN: 
Moves: e4 d6 d4 Nf6 Nc3 g6 Be3 Bg7 Qd2 c6 f3 b5 Nge2 Nbd7 Bh6 Bxh6 Qxh6 Bb7 a3 e5 O-O-O Qe7 Kb1 a6 Nc1 O-O-O Nb3 exd4 Rxd4 c5 Rd1 Nb6 g3 Kb8 Na5 Ba8 Bh3 d5 Qf4 Ka7 Rhe1 d4 Nd5 Nbxd5 exd5 Qd6 Rxd4 cxd4 Re7 Kb6 Qxd4 Kxa5 b4 Ka4 Qc3 Qxd5 Ra7 Bb7 Rxb7 Qc4 Qxf6 Kxa3 Qxa6 Kxb4 c3 Kxc3 Qa1 Kd2 Qb2 Kd1 Bf1 Rd2 Rd7 Rxd7 Bxc4 bxc4 Qxh8 Rd3 Qa8 c3 Qa4 Ke1 f4 f5 Kc1 Rd2 Qa7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 Nc6 Be3 Qc7 f4 Be7 Qf3 O-O O-O-O Nxd4 Rxd4 b5 g4 e5 Nd5 Nxd5 Rxd5 b4 f5 a5 Ba4 Bb7 g5 Rfc8 Qg2 Bxd5 exd5 Qc4 Kb1 Qh4 Bc6 a4 Rf1 Qc4 Rg1 b3 a3 Rxc6 dxc6 Rc8 Rd1 Rxc6 cxb3 axb3 Ka1 e4 Qd2 f6 h4 Kf7 Qd5 Qxd5 Rxd5 g6 fxg6 Kxg6 Rb5 fxg5 hxg5 Rc2 Kb1 Rg2 Rxb3 Bxg5 Bxg5 Kxg5 a4 Kf4 Rb7 e3 Re7 Kf3 Kc1 d5 a5 d4 Rxh7 d3 Rh1 Rc2 Kb1 e2 a6 Rc8
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7 Nxd7 Bxd7 O-O Bd6 c4 c6 cxd5 cxd5 Qh5 O-O Qxd5 Bc6 Qh5 g6 Qh3 Ng5 Qg4 Ne6 Bh6 Re8 Nc3 Bf4 Bxf4 Qxd4 Be4 f5 Qd1 Qxd1 Rfxd1 fxe4 Be3 Kf7 b4 b6 a4 Red8 h3 Rd3 Rdc1 Ke7 b5 Bb7 a5 bxa5 Rxa5 a6 bxa6 Rxa6 Rxa6 Bxa6 Nxe4 Bb7 Bc5 Kd8 Nf6 h5 Bb4 Rd4 Re1 Bc8 Bc3 Rd3 Be5 Bb7 Bg3 Ke7 Ng8 Kd7 Nh6 Bc6 Nf7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 hxg4 O-O e6 Qd2 Nd7 Bxd6 Ne5 Ncb5 axb5 Nxb5 f6 Rfd1 Kf7 Qe2 Bd7 Bxe5 fxe5 Nd6 Kg6 Rd3 Qf6 Qxg4 Qf4 h3 Qxg4 hxg4 Bc6 c4 Rhd8 f3 Bf8 c5 b6 b4 bxc5 bxc5 Ra5 Rc3 Rb8 Rc2 Kf6 Kf2 Ba4 Rc3 Rb2 Kg3 Rc2 Rc1 Rxa2 Rb1 Bc6 Rb6 Ra6 Rb8 Be7 Rc1 Ra8 Rb6 R2a6 Rbb1 Be8 Rh1 Bg6 Rb7 Rc6 Nf5 Bf8 Rh8 exf5 gxf5 Bf7 Kg4 Kg7 Rh5 Bxc5 Rxg5 Kf6
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 g3 Bg7 Bg2 d5 cxd5 Nxd5 e4 Nb6 Ne2 O-O O-O Nc6 d5 Na5 Nbc3 c6 b3 cxd5 exd5 Bg4 Bd2 e6 dxe6 Bxe6 Ne4 Nc6 Bg5 Qxd1 Raxd1 h6 Be3 Rad8 Nd6 Bg4 f3 Bc8 Bc5 Bf6 f4 Bg4 Ne4 Bxe2 Nxf6 Kg7 Rxd8 Rxd8 Re1 Rd2 Ne8 Kg8 Nd6 Nd7 Be3 Rxa2 Bd5 Nd8 b4 Rb2 f5 gxf5 Nxf5 Ne6 Bc1 Rb1 Be4 Ra1 Ne7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 Qc7 Qc2 dxc4 Bxc4 e5 Bd3 Re8 e4 exd4 cxd4 Bg4 e5 Bxf3 exf6 Nxd4 Bxh7 Kh8 fxg7 Kxg7 Bb2 Rad8 gxf3 Rh8 Kh1 Rxh7 Rg1 Kh8 Rg3 Qe5 Rag1 Rh4 Qc1 Kh7 Qb1 Kh8 Qf1 Qe6 Qg2
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 Nf3 a6 e3 e6 Bxc4 c5 O-O Nf6 Bb3 cxd4 exd4 Nc6 Nc3 Be7 Re1 O-O Bf4 Na5 Bc2 b5 d5 exd5 Qd3 Nc6 Bc7 Qd7 Ne5 Nxe5 Bxe5 g6 Bxf6 Bxf6 Nxd5 Bg7 a4 bxa4 Bxa4 Qd8 Bc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 a6 e5 b5 Qb3 Nfd7 e6 fxe6 Be3 Nb6 h4 Nc6 h5 Rxf3 gxf3 Nxd4 Rd1 c5 Bxd4 cxd4 hxg6 h6 Rh5 Qe8 Ne2 Qxg6 Rh1 Kh8 Rg1 Qf7 Nxd4 Nd5 Qd3 Bd7 Qe4 Rc8 Bd3 Nf6 Qh4 Rc5 Ne2 Nd5 Rg6 Qf8 Qe4 Qf7 Kd2 Nf6 Qe3
Result: 1-0
Processing game...
FEN: 
Moves: Nf3 d5 d4 Nf6 c4 c6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6 e4 c5 e5 cxd4 Nxb5 Nxe5 Nxe5 axb5 Bxb5 Bd7 Nxd7 Qa5 Bd2 Qxb5 Nxf8 Rxf8 a4 Qc4 b3 Qd3 Qf3 Qe4 Qxe4 Nxe4 b4 Ke7 f3 Nc3 Bxc3 dxc3 Ke2 Rfd8 Rhd1 c2 Rdc1 Rd4 Rxc2 Rxb4 a5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O d3 d6 c3 Na5 Bc2 c5 Nbd2 Nc6 Nf1 Re8 h3 h6 d4 cxd4 cxd4 exd4 Nxd4 Nxd4 Qxd4 Bb7 Ng3 d5 e5 Ne4 Nxe4 dxe4 Be3 Qxd4 Bxd4 Bb4 Red1 Bd5 a3 Ba5 Bc5 Bb7 Bd6 Bb6 Bb3 Rac8 a4 g5 axb5 axb5 Bb4 Rcd8 Bc3 e3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Qd2 Nc6 Nb3 Nge5 f3 b5 Bf2 Rb8 Nd4 b4 Nd1 Nxd4 Bxd4 d5 exd5 Qxd5 c3 O-O Be2 Rd8 Qe3 Bb7 Nf2 bxc3 Bxc3 Qe6 Kf1 Bd5 b3 Rbc8 Rd1 Rxc3 Qxc3 Ng4 Qa5 Ne3 Kg1 Rc8 Qxa6 Nxd1 Qxe6 Bxe6 Bxd1 Rc1
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Nc6 O-O Be6 Bb5 Bc5 Nbd2 Qd6 e5 Qd5 Ng5 O-O-O Bc4 Qd7 Nxe6 fxe6 b4 Nxb4 Qb3 Nd5 Ne4 Bb6 a4 a5 Nd6 Kb8 Bxd5 exd5 Bd2 cxd6 Qxb6 dxe5 f4 Nf6 fxe5 Ne4 Bxa5 d3 e6 Qd6 Qxd6 Rxd6 e7 Rf6 Rxf6 Nxf6 Rd1 Re8 Bb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 Qxd5 Nf3 Qf5 Qxf5 exf5 a3 Be7 Bg5 Be6 e3 c6 Bd3 Nbd7 O-O h6 Bh4 a5 Rac1 O-O Ne2 g5 Bg3 Ne4 Nc3 Nxc3 Rxc3 Nf6 Rcc1 Rfd8 Rfd1 Rac8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nc3 d6 g3 Nc6 Bg2 g6 d3 Bg7 Be3 Nf6 Nge2 O-O h3 e5 O-O b5 Nxb5 Rb8 a4 a6 Na3 Rxb2 Nc4 Rb8 f4 exf4 Nxf4 Na5 Nd2 Bd7 Ra2 Bc6 Nf3 Ba8 c4 Nd7 Raf2 Nb3 h4 Nd4 Bh3 Nxf3 Qxf3 Ne5 Qd1 Qe7 h5 Rb4 h6 Bh8 Ne6 Rfb8 Ng5 Rb2 Bf4 Bf6 Nf3 Bc6 Nxe5 Bxe5 Qf3 f5 Bxe5 Qxe5 g4 Rxf2 Rxf2 Rb1 Bf1 Rb2 Rxb2 Qxb2 gxf5 Qd4 Kh1 Bxa4 fxg6 hxg6 e5 Qxe5 Qb7 Be8 Bg2 Qh5 Kg1 Qxh6 Qe7 Qc1 Bf1 Bf7 Qxd6 Qe3 Kh1 a5 Qd8 Be8 Bg2 Qe1 Kh2 Qe5 Kh3 Kg7 Bd5 a4 Qb6 Bd7 Kg2 Qe2 Kg1 a3 Qxc5 Qe1 Kg2 a2 Qd4 Kh7
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 Nc3 Nc6 d4 cxd4 Nxd4 Nf6 Nxc6 bxc6 e5 Nd5 Ne4 Bb7 Be2 c5 O-O Qc7 Nd6 Bxd6 exd6 Qc6 f3 c4 Qd4 O-O Bxc4 Qxd6 Bb3 Qb6 Rd1 Rfc8 Qxb6 Nxb6 a4 d5 a5 Nc4 a6 Bc6 Bxc4 dxc4 Be3 Bd5 Ra5 Rc6 Rda1 f6 h4 Kf7 Rb5 Rc7 Kf2 Rd8 Raa5 Ke7 Kg3 h5 b4 cxb3 cxb3 Rg8 Rc5 Rd7 b4 g5 Rc2 g4 Kf2 g3 Ke1 e5 Rd2 Rgd8 Rc5 Ke6 b5 Rb8 Rd3 Rbd8 Rd2 Rb8 Rd3 Rbd8 b6 axb6 Rb5 Bc4 Rxb6 Kf5 Rxd7 Rxd7 a7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f3 e6 Be3 b5 g4 h6 Qd2 Nbd7 O-O-O Bb7 h4 b4 Na4 Qa5 b3 Nc5 a3 Nxa4 axb4 Qc7 bxa4 d5 e5 Nd7 f4 Nb6 a5 Nc4 Qc3 Qe7 Bxc4 dxc4 Bd2 Bxh1 Rxh1 Qb7 Rd1 Be7 Qf3 O-O-O Qc6 Qxc6 Nxc6 Rd7 Nxe7 Rxe7 Re1 h5 g5 Kb7 Be3 Rd7 Bc5 Kc6 Bd6 f6 gxf6 gxf6 Rg1 f5 Kd2 Kd5 Ke3 Rhh7 Bf8 Rhf7 Bh6 Rh7 Rg6 Rb7 Bf8 Rhf7 Bd6 Rg7 Rg5 Rbf7 c3 Kc6 Kf3 Kb5 Bc5 Ka4 Bd4 Rd7 Ke3 Kb3 Ke2 Rxg5 fxg5 Rxd4 cxd4 c3 g6 c2 g7 c1=Q g8=Q Qc4 Ke3 Kc3 Qd8 Qd3 Kf4 Qd2 Kf3 Qd1 Ke3 Qg1 Ke2 Qg2 Ke3 f4
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Qxd4 Nc6 Bb5 Bd7 Bxc6 Bxc6 Nc3 Nf6 Bg5 e6 O-O-O Be7 Rhe1 O-O Kb1 h6 Bh4 Re8 Bg3 d5 e5 Ne4 Nxe4 dxe4 Qxd8 Rexd8 Nd4 Be8 c3 Rac8 Kc2 b5 Rxe4 b4 Re3 a5 Ne2 Bc6 f3 Rxd1 Kxd1 Bc5 Rd3 Bb5 Rd2 Be3 Rd6 bxc3 Nxc3 Bf1 Bh4 g5 Be1 Bxg2 Ke2 Bf4 Bg3 Bc1 Na4 h5 Rd1 h4 Be1 Bf4 Bc3 Kg7 Nb6 Rh8 Kf2 h3 Bd2 Bxh2 Nd7 Kg6 Be3 g4 fxg4 Bc6
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 dxc4 Nf3 e6 e3 c5 Bxc4 a6 O-O Nf6 Bb3 Nc6 Nc3 cxd4 exd4 Be7 Re1 O-O a3 Na5 Bc2 b5 d5 Nc4 Qd3 Re8 a4 exd5 axb5 a5 b3 Nd6 Nd4 Bb7 f3 Rc8 Na4 Bf8 Bg5 g6 Qd2 Rxe1 Rxe1 Nde8 Re2 Bb4 Qe3 Rc7 Bd3 Re7 Qc1 Rxe2 Bxe2 Qe7 Qe3 Qxe3 Bxe3 Nd7 Nc6 Bxc6 bxc6 Nb8 Bb6 Bd6 Nc3 Bc7 Bf2 d4 Nd5
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f3 e6 Be3 b5 g4 h6 Qd2 Nbd7 O-O-O Bb7 h4 b4 Nb1 d5 Bh3 g5 hxg5 hxg5 exd5 Nxd5 Bxg5 Qb6 Bg2 Rxh1 Bxh1 Rc8 Re1 Qa5 f4 Qxa2 f5 Nc5 fxe6 Bg7 exf7 Kxf7 Bxd5 Qxd5 Re7 Kg8 Rxg7 Kxg7 Nc3 bxc3 Nf5 Kf7 Qxd5 Bxd5 Nd6 Kg6 Nxc8 Kxg5 Nb6 Be6 bxc3 Kxg4 Kb2 Kf4 Ka3 a5 Na4 Ne4 Nb2 Nxc3 Nd3 Ke3 Nc5 Bf5 Kb2 Nd5 Nb7 a4 c4 Nb6 Nd6 Bd3 c5 Nd5 Ka3 Bc2 Nb5 Ne7 Na7 Kd4 c6 Nd5 Nb5 Kc5 c7 Bf5
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O h3 Bb7 d3 d6 a3 Na5 Ba2 c5 Nc3 Nc6 Bg5 Qd7 Nh2 Ne8 Bd2 Nc7 Nf1 Kh8 Ng3 Nd4 Nce2 Nde6 b4 d5 bxc5 Bxc5 Bb4 Rfe8 Bxc5 Nxc5 Nc3 Rad8 Qh5 f6 d4 exd4 Nxd5 Re5 Qh4 Nxd5 exd5 Bxd5 Rxe5 fxe5 Bxd5 Qxd5 Re1 Ne6 Nf5 Nf4 Qg5 Rd7 Nh4 h6 Qg4 g5 Nf3 e4 Rxe4 Qxe4 Qxd7 d3 cxd3 Qxd3 Qc8 Kg7 Qb7 Kg8 Qxa6 Ne2 Kh2 Qe4 Qf6 Qf4 Qxf4 gxf4 g4 fxg3 fxg3 Nc3 Nd4 h5 h4 Kf7 Kh3 Kf6 g4 hxg4 Kxg4 Kg6 h5 Kh7 Kh4 Kg8 h6 Kh7 Kh5 Ne4 Nxb5 Nf6 Kg5 Ne4 Kf5 Nc5 Ke5 Kxh6 Kd4 Na6 Kd5 Kg6 Nd4 Kf6 Kd6 Kf7 Ne6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 g4 b5 g5 b4 Nd5 Nxd5 exd5 Bf5 Bd3 Bxd3 Qxd3 Be7 Rg1 O-O O-O-O a5 Nd2 f5 gxf6 Nxf6 Kb1 Nh5 Rg4 Qd7 Rc4 Qf5 Qxf5 Rxf5 Rc7 Bf8 Rb7 Nf6 Rb5 Ne8 Rb7 Nf6 Rb5 Ne8 Rb7 Nf6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O O-O d6 Nc3 e5 d4 Nbd7 h3 exd4 Nxd4 Re8 e4 a6 Re1 Rb8 a4 c6 Nc2 Ne5 Ne3 Be6 b3 Nfd7 Bd2 Nc5 Rf1 Ned3 Rb1 a5 f4 f5 exf5 gxf5 Kh2 Bd7 Nc2 Ne6 Na2 Ndc5 Bc3 Bxc3 Nxc3 Qf6 Qd2 Ra8 Rfe1 Nc7 Rxe8 Rxe8 b4 axb4 Nxb4 Rd8 Nc2 Be6 Ne3 Bf7 g4 Bg6 g5 Qg7 Ne2 N7e6 a5 Qc7 Rd1 Ra8 Ng3 Qxa5 Qf2 Ng7 Rxd6 Re8 h4 Re6 Rd1 Qc7 Nexf5 Nxf5 Qxc5 b6 Qf2 Nxh4 Bh3 Rd6 Rxd6 Qxd6 Bg4 Bd3 f5 h6 g6 h5 c5 bxc5 Qa2 c4 Qa8 Qf8 Qa7 Nxg6 fxg6 Bxg6 Be6 Bf7 Bf5 Qd6 Kh3 Bg6 Qa8 Kg7 Qb7 Kg8 Qc8 Kg7 Qb7 Kg8 Qa8 Kg7 Qa5 Qf4 Qa7 Kh6 Qc5 Bxf5 Nxf5 Kg6 Nh4 Kf7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Bg5 Ne4 Bf4 Nxc3 bxc3 Bg7 e3 O-O Nf3 c5 cxd5 cxd4 cxd4 Qxd5 Be2 Nc6 O-O Bf5 Qa4 Qa5 Qb3 Qb4 Rfc1 Rac8 Bb5 Be6 Qd3 Qa5 a3 Bd5 Rab1 Ba2 Ra1 Bd5 Rab1 Ba2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 hxg4 O-O Nc6 Nf5 Bxc3 bxc3 Qa5 Qxg4 f6 Rab1 Qxc3 Rfd1 Ne5 Bxe5 Qxe5 g3 b5 c4 bxc4 Qf3 c3 Rbc1 Bb7 Qxc3 Bxe4 Nxd6 exd6 Re1 Qxc3 Rxc3 d5 f3 Kf7 fxe4 Rhe8 Rc7 Kg6 Rf1 dxe4 Rc6 Rf8 Re6 Rae8 Rfxf6 Rxf6 Rxe8 Kf5 Kf2 Rh6 Rf8 Ke5 Re8 Kf5 Rf8 Ke5 Re8
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 O-O Bg2 cxd4 Nxd4 d5 cxd5 Nxd5 Bd2 Bxc3 bxc3 Nb6 Be3 Nd5 Qd2 Nd7 Bg5 Qc7 Nb5 Qc5 c4 Qxc4 Rb1 N7b6 O-O h6 Bxh6 gxh6 e4 Ne7 Rfc1 Qa4 Qxh6 Bd7 Rc5 Ng6 Rg5 Qc2 Na3 Qd3 h4 Qxa3 h5 Qe7 e5 Be8 Be4 f5 exf6 Rxf6 hxg6 Qg7 Qh7 Kf8 Qh4 Rc8 Rh5 Bxg6 Rh8 Kf7 Rxc8 Nxc8 Rxb7 Ne7 Bxg6 Qxg6 Qb4 Qf5 Qxe7 Kg6 Qh7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O Bf4 Na6 e4 c5 dxc5 Be6 Qb5 Bd7 Qxb7 Nxc5 Qb4 Ne6 Be5 a5 Qa3 Qb6 Bc4 Rfc8 Bxf6 Bxf6 Nd5 Qb8 Nxf6 exf6 Bxe6 Bxe6 O-O Qb4 Qxb4 axb4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 dxc4 e4 b5 a4 b4 Na2 Nf6 e5 Nd5 Bxc4 e6 Nf3 Be7 Bd2 a5 Nc1 Nd7 Nb3 Bb7 O-O h6 Qc2 Rc8 Qe4 c5 Rac1 Kf8 Bb5 Ba8 Qd3 g5 Bxd7 Qxd7 dxc5 Qxa4 Nfd4 Qe8 c6 Bxc6 Nxc6 Rxc6 Nxa5 Rxc1 Rxc1 Kg7 Nc6 Qd7 Bxb4 Rc8 Bxe7 Rxc6 Bf6 Kg8 Rxc6 Qxc6 Qd2 Qa4 h3 Qe4 f3 Qf5 Qa5 Nxf6 Qd8 Ne8 Qxe8 Kg7 Qb5 Qc2 Qb4 Qe2 Qd4 h5 b4 Qe1 Kh2 h4 b5 Qg3 Kh1 Qe1 Qg1 Qxe5 Qb1 Qd5 b6 Qb7 Qb4 e5 Qc5 Kg6 Qc7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e5 Nf3 Nbd7 a4 Be7 Bc4 Qa5 Qe2 O-O O-O exf4 Bxf4 Ne5 Nd5 Nxd5 Bxd5 Be6 Bxe6 Nxf3 Qxf3 fxe6 Qb3 Qc5 Kh1 Qc8 Rad1 b5 axb5 axb5 Qxb5 Rb8 Qd3 Rxb2 Bxd6 Bxd6 Qxd6 Rxf1 Rxf1 Rxc2 h3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 e3 a6 Qc2 b5 b3 Bg4 Nge2 Nbd7 Nf4 e5 dxe5 Nxe5 h3 Be6 Be2 Bd6 O-O O-O Bb2 Ng6 Nxg6 hxg6 cxd5 cxd5 Bf3 Rc8 Qd2 Bb8 Rfd1 Qc7 g3 Rfd8 Rac1 Qd7 h4 Ba7 Ne2 Rxc1 Qxc1 Bg4 Bxg4 Qxg4 Bxf6 gxf6 Rd2 Re8 Qd1 Qf3 Nf4 Qxd1 Rxd1 d4 exd4 Rd8 d5 Bb8 Ne2 Kg7 Kg2 f5 Kf3 Bd6 Nf4 Rd7 Ng2 Bb8 Ne3 Ba7 h5 Bc5 a4 bxa4 bxa4 Bxe3 Kxe3 gxh5 Kd4 Kf8 Rb1 f4 a5 fxg3 fxg3 Ke7 Rb6 Rd6 Kc5 Rg6 d6 Kd7 Rb7 Kc8 Rc7 Kd8 Ra7 Rxg3 Rxa6 Rc3 Kb4 Rc1 Ra8 Kd7 a6
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Be3 Qb6 Nh4 Bd7 Rab1 Rfe8 b4 Rxe3 fxe3 cxb4 Qf4 Qc5 Ne4 Nxe4 Qxe4 Re8 Qf4 Qxe3 Qxe3 Rxe3 Bxa6 bxa6 Rxb4 Rd3 Rbf4 Rxd5 Rxf7 Bb5 Rxg7 Kxg7 Rc1 g5 Nf3 g4 Nh4 Kf6 Rc7 h5 Rxa7 Rd1 Kf2 Rd2 Ke3 Rxa2 Rb7 Re2 Kf4 Re4 Kg3 Re3 Kf4 Re4 Kg3 Re7 Rb6 Re6 Rb7 Bd3 Rd7 Bb5 Rh7 Be8 Ra7 Re3 Kf4 Ra3 Rb7 Ra4 Ke3 Re4 Kd3 Re7 Rb2 Bb5 Kd4 Re2 Rb3 Kg5 g3 Rd2 Ke4 Rc2 Ke5 Rxh2 Rc3 Re2 Kd4 Rd2 Ke4 Re2 Kd4 Rd2 Ke4 Kf6 Rc5 Re2 Kf4 Bd3
Result: 0-1
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Nb6 Nc3 Qe6 Qe4 Bb4 Bd2 Ba6 b3 Bxc3 Bxc3 d5 Qh4 dxc4 Be2 Nd5 Bd4 c5 Bxc5 Nc3 Bxc4 Qxe5 Be3 Ne4 O-O Bxc4 bxc4 O-O Rfe1 Rfe8 f3 Nd6 Bf2 Qf5 c5 Nb5 Qb4 Qd3 Red1 a5 Qa4 Qe2 Re1 Qd3 Rxe8 Rxe8 Rd1
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 Bb5 Bd7 Bxd7 Qxd7 c4 Nc6 Nc3 Nf6 O-O g6 d4 cxd4 Nxd4 Bg7 Nde2 Qe6 Nd5 Qxe4 Nc7 Kd7 Nxa8 Qxc4 Nb6 axb6 Nc3 Ra8 a4 Ne4 Nxe4 Qxe4 Qb3 f5 Bg5 Qb4 Qf7 Be5 h3 Rxa4 Rxa4 Qxa4 Qxh7 Bxb2 Qxg6 Qe4 Qf7 Bd4 Qb3 f4 Qf7 Be5 h4 b5 h5 Qc4 Qf5 Qe6 Qxe6 Kxe6 g3 fxg3 fxg3 b4 Bf4 Bd4 Kh1 b3 g4 Kd5 g5 e6 h6 Ne7 Rd1 e5 Be3 Kc4 Bxd4 exd4 Kg2 b2 Kf3 Kc3 h7 Ng6 Ke4 Kc2 Rh1 d3 Kf5 b1=Q Rxb1 Kxb1 Kxg6 d2 h8=Q d1=Q Qh7 b5 Kf6 Kb2 Qh2 Ka1 Qf4 b4 Qxb4 Qf3 Kg7 d5 Qd4 Kb1 g6 Qe4 Qg1 Kb2 Qf2 Kc1 Kf6 d4 g7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 g6 c4 Bg7 g3 d5 Bg2 O-O O-O dxc4 Na3 c3 bxc3 c5 e3 Nc6 Qe2 Bf5 Bb2 e5 Nc4 e4 Nfe5 Nxe5 Nxe5 h5 h3 Nd7 Nxd7 Qxd7 Rfd1 cxd4 cxd4 Bxh3 Bxe4 h4 Bf3 hxg3 fxg3 Rfe8 e4 Rac8 Kh2 Bf5 Qg2 Bg4 Rd2 g5 Rf1 Rc6 Bxg4 Qxg4 e5 Rh6 Kg1 f6 Qf3 Qxf3 Rxf3 fxe5 dxe5 Rg6 Rf5 g4 Rd7 Rc8 Bd4 Rc2 Rf2 Rc4 Rd2 Bh6 Re2 Bg7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Nc3 d6 d4 cxd4 Nxd4 Nf6 Bg5 e6 Qd2 a6 O-O-O h6 Be3 Be7 f3 Nxd4 Qxd4 b5 Kb1 Bb7 h4 Rc8 Be2 d5 exd5 Nxd5 Nxd5 Bxd5 Qxg7 Bf6 Qg4 Bxa2 Kxa2 Qa5 Kb1 Bxb2 Kxb2 Qc3 Ka2 Qxc2 Ka1 Qc3 Ka2 Qc2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Rd1 Qb6 d6 Rad8 Na4 Qc6 Be3 Ne4 Qb5 Bd7 Qxc6 Bxc6 Bxa6 Bxa4 Bxb7 Bxd1 Bxe4 Bxf3 Bxf3 Bxb2 Rd1 Bd4 Bxd4 cxd4 Rxd4 Rd7 h4 Rfd8 Ra4 Rb8
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 Nf3 d6 Nd2 Nbd7 f3 d5 e3 Re8 cxd5 exd5 Bb5 c6 Be2 Qe7 b4 h6 Bxf6 Nxf6 Kf2 Qd6 Nf1 a5 Ng3 Ba6 Bxa6 Rxa6 Rhc1 b5 bxa5 Rea8 Nf5 Qe6 g4 h5 h3 g6 Ng3 hxg4 hxg4 Kg7 g5 Nd7 e4 Rxa5 Qxc6 Qxc6 Rxc6 b4 Rd6 R8a7 a4 b3 Rb1 Rb7 Rxd5 Rxa4 Ne2 Ra2 Ke1 Nf8 Nc1 Rc2 Rc5 Rxc5 dxc5 Ne6 Rxb3 Rc7 Rc3 Rxc5 Rxc5 Nxc5 Ne2 Ne6 f4 f6 gxf6 Kxf6 Kf2 Nc5 Ke3 g5 f5 Ke5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2 cxd4 cxd4 Qa5 Bd2 Qxa2 O-O Bg4 Be3 Nc6 d5 Na5 Bg5 Qa3 Bd2 Bc3 Bc1 Qd6 e5 Qc7 d6 exd6 exd6 Qd7 Bf4 Rfe8 h3 Bf5 Bb5 Nc6 Rc1 Bf6 Qd2 a6 Bxc6 bxc6 Nd4 Be4 Nb3 Bd5 Nc5 Qa7 Qa5 Bd4 d7 Red8 Bc7 Bxc5 Bxd8 Bd6 Bf6 Qxd7 Bb2 Rf8 Rfe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nxc6 Qf6 Qd2 dxc6 Nc3 Ne7 Qf4 Qd6 Be2 Ng6 Qg3 f5 f4 Qd4 Bd3 Bb4 exf5 Bxc3 bxc3 Qxc3 Kf2 O-O fxg6 Qxa1 Qh4 h6 Bc4 Kh8 Bf7 Qc3 Be3 Be6 Bxe6 Rae8 Qh3 Qxc2 Kf3 Qxg6 Bg4 Qe4 Kf2 Rxf4 Bxf4 Qxf4 Bf3 Qd4 Kg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 c6 e3 Bf5 Bd3 Bg6 Bxg6 hxg6 Nbd2 e6 Qc2 Nbd7 b3 a5 O-O Be7 e4 dxe4 Nxe4 Nxe4 Qxe4 a4 Bd2 O-O Bc3 Qc7 h3 Rfd8 Rfd1 Bf6 Qc2 axb3 axb3 Qf4 Qb2 b5 Qe2 bxc4 bxc4 Rxa1 Bxa1 g5 g3 Qf5 Kg2 g4 hxg4 Qxg4 Rb1 Nf8 Re1 Ra8 Ng1 Qxe2 Nxe2 Ra4 Rc1 Nd7 c5 g5 Bc3 Ra2 Kf1 Nf8 Rb1 Ng6 Ra1 Rxa1 Bxa1 g4 Kg2 Ne7 f3 gxf3 Kxf3 Nd5 Bb2 Kh7 Bc1 Kg6 Bf4 Bg7 Bd6 f5 g4 Bh6 Bf4 Bg7 Be5 Bh6 Bf4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 Nc3 d6 d4 cxd4 Nxd4 Nf6 Bg5 e6 Qd2 a6 O-O-O h6 Be3 Bd7 f3 b5 Nxc6 Bxc6 Ne2 Qc7 Kb1 Bb7 h4 Rc8 Rh3 h5 Nd4 Nd7 Bg5 g6 Rh1 d5 exd5 Bxd5 Bf4 Qb7 Nb3 Bg7 Bd6 Qc6 Ba3 Nb6 Bd3 Nc4 Bxc4 bxc4 Nd4 Qb6 Qg5 Bf8 Bxf8 Rxf8 Rhe1 Rb8 b3 Rd8 Ne2 cxb3 axb3 Rd7 Kb2 Qd8 Qf4 Qc7 Rxd5 exd5 Nd4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Be2 c5 d5 e6 O-O exd5 exd5 Bf5 Rd1 Qb6 Qh4 Bc2 Rd2 Bf5 Rd1 Bc2 Rd2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 Bg5 Bb7 e3 d6 f3 h6 Bh4 Nbd7 Nh3 c5 Be2 d5 dxc5 bxc5 O-O a5 Rad1 Qb6 Rd2 Bc6 Bf2 Ba4 Nf4 Bb3 cxd5 Nxd5 Nxd5 exd5 Bd1 Bc4 Re1 Qe6 b3 Ba6 Qxa5 Bb7 Qc7 Bc6 a4 Rfc8 Qg3 c4 e4 dxe4 fxe4 Ne5 Bd4 Re8 Rc2 Ng6 Rxc4 Bd5 Bg4 Qe7 Rc7 Qb4 Qc3 Rab8 Qxb4 Rxb4 Bc3 Rbxe4 Rxe4 Bxe4 Rc8 Rxc8 Bxc8 Bc2 a5 Bxb3 a6 Nf4 Bb7
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nf3 d5 Nc3 c6 Bg5 Nbd7 cxd5 exd5 e3 Be7 Bd3 O-O Qc2 Re8 O-O Nf8 Rab1 a5 a3 Ng6 b4 Bd6 Rfe1 axb4 axb4 Bg4 Nh4 Bd7 h3 Ra3 e4 Bf4 Nxg6 Bxg5 Ne5 dxe4 Nxe4 Nxe4 Bxe4 g6 Nc4 Ra7 b5 Bf4 bxc6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 Qxd5 e3 c5 Bd2 Bxc3 Bxc3 cxd4 Bxd4 Nc6 Bc3 Qc5 Rc1 O-O Nf3 Nd5 Bd2 Qxc2 Rxc2 e5 Rc5 Rd8 Bc4 Nb6 Bb3 Re8 Ng5 Re7 f3 Nd7 Rc1 Nf6 Bc2 h6 Ne4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Qh5 Nc6 Bc4 Qe7 Nf3 Nf6 Qh4 d6 d3 h6 h3 Be6 Nc3 Bxc4 dxc4 Nd4 Nxd4 exd4 Ne2 c5 f3 d5 cxd5 Nxd5 Qxe7 Nxe7 Bd2 O-O-O O-O-O g6 Nf4 Bg7 c4 dxc3 Bxc3 Bxc3 bxc3 b6 c4 Nc6 Kb2 Rhe8 Rxd8 Rxd8 Nd5 h5 a4 Kd7 Kc3 Ke6 f4 Nd4 Rd1 Ne2 Kc2 Nd4 Kc3 Ne2
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 d6 Bb3 O-O Bg5 Be6 Nbd2 a6 h3 Ba7 Bh4 Kh8 g4 Ne7 Bxf6 gxf6 Nh4 Ng6 Ng2 c6 Qf3 d5 Nf1 a5 Ng3 Bc5 a4 Be7 Ba2 Ra6 Nh5 Rb6 Qe2 Qd6 O-O Rd8 Rfd1 d4 Rd2 Qc5 Rc2 Qd6 Ne1 Rg8 Nf3 Qd7 Kh1 c5 Bxe6 fxe6 Nd2 Ra6 Nc4 Bd8 Rcc1 Rf8 f3 Bc7 Ra3 Raa8 Rb3 Qxa4 Rxb7 Qc6 Rb3 f5 Nd2 Rf7 c4 a4 Rb5 Ba5 Nf1 Bb4 Nfg3 fxg4 fxg4 Raf8 Rf1 Qe8 Rxf7 Qxf7 g5 Nf4 Qf3 Nxh5 Qxh5 Qf2 Ne2 Qf3 Qxf3 Rxf3 Kg2 Rxd3 Ng3 Kg7 Rb6 Be1 Nf1 Kf7 Nh2 Rd2 Kh1 Re2 Ng4 Rxe4 Nf6 Re2 Nxh7
Result: 0-1
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nf3 e6 Nbd2 f5 g3 Nf6 Bg2 Bd6 O-O O-O Ne5 Nbd7 Ndf3 Ne4 Bf4 Nxe5 Bxe5 Qe7 Qc2 Rd8 c5 Bxe5 Nxe5 Ng5 Nd3 Qf6 e3 Bd7 b4 Nf7 f4 Be8 a4 a6 Qc3 h6 Bf3 Kh7 Rf2 Nh8 h4 Qe7 h5 Bd7 Rg2 Rg8 Kf2 Nf7 Ke2 Qd8 Kd2 Qc7 g4 a5 g5 hxg5 fxg5 axb4 Qxb4 Kh8 Be2 Ra5 g6 Nh6 Qb6 Qxb6 cxb6 Rga8 Nc5 Bc8 Kc3 Rb8 Kb4 Raa8 a5 Kg8 Kc3 Kf8 Bf3 Ke7 Rf2 Bd7 a6 bxa6 b7
Result: 1-0
Processing game...
FEN: 
Moves: d4 f5 Bg5 Nf6 Bxf6 exf6 e3 d5 c4 Be6 cxd5 Bxd5 Nc3 Bf7 Bd3 g6 h4 Bg7 h5 Nd7 Nge2 a6 Nf4 Nb6 Qf3 c6 Qg3 g5 Bxf5 Bh6 Ne6 Bxe6 Bxe6 Bf8 Qf3 Bb4 O-O Bxc3 bxc3 Rb8 e4 Qd6 Bb3 Nd7 Rfe1 Kd8 e5 fxe5 dxe5 Qe7 Qg3 Rc8 Rad1 Ke8 Rd6 Nc5 Rf6 Nxb3 axb3 Rg8 Qg4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 d5 cxd5 Nxd5 Nf3 Nf6 Nc3 Nbd7 e4 e6 Bd3 Be7 O-O a6 Re1 b6 e5 Nd5 Nxd5 exd5 Qc2 c5 e6 Nf6 exf7 Kf8 Ne5 Qc7 Bf4 Bd6 Qe2 Bd7 Bf5 Bc6 Qf3 c4 Qh3 Rd8 Ng6 hxg6 Qxh8 Kxf7 Be6 Ke7 Qxg7
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 e6 e4 Ne7 Nc3 Ng6 Bd3 Be7 O-O O-O Re1 c6 Bc2 Nd7 e5 Nb6 Ne4 Nh4 Nxh4 Bxh4 Re3 Nd5 Rh3 f5 exf6 Bxf6 Qh5 Bxd4 Bg5 Nf6 Nxf6 Bxf6 Bxh7 Kh8 Bg6 Kg8 Qh8
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 d6 d4 Bd7 dxe5 Nxe5 Nxe5 dxe5 Qh5 Ke7 Qxe5 Kf7 Bc4 Kg6 Qg3 Kf5 Qf3 Ke5 Bf4
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 d6 Bb3 Bg4 h3 Bh5 Nbd2 O-O Qe2 Qd7 g4 Bg6 Nf1 Rfe8 Ng3 Ne7 Bg5 h6 Bxf6 gxf6 O-O-O Red8 Nh4 Bh7 Nh5 Qc6 Nxf6 Kf8 Nxh7 Kg7 g5 Kxh7 Qh5 Ng8 Qxf7 Kh8 Ng6
Result: 1-0
Processing game...
FEN: 
Moves: e4 e5 Bc4 Nf6 d3 Be7 Nc3 Nc6 f4 d6 Nf3 Bg4 O-O Nd4 fxe5 dxe5 Bxf7 Kf8 Bb3 Nxf3 gxf3 Bh3 Re1 Bc5 Kh1 Nd7 Be3 Bxe3 Rxe3 Qg5 Re2 Qf4 Rf2 Qe3 Qg1 Nf6 Re1 Qh6 Nd5 Nxd5 Bxd5 c6 Bb3 Rd8 Qg3 Re8 f4 exf4 Rxf4 Ke7 Rf7 Kd8
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 Bd2 Bg7 e4 Nb6 Be3 O-O Be2 Nc6 Nf3 Bg4 d5 Bxf3 gxf3 Na5 Bd4 Qd6 Bxg7 Kxg7 f4 Qf6 Qd2 c6 dxc6 Rfd8 Qe3 Nac4 Qc5 Rac8 c7 Rd7 Bxc4 Rcxc7 Qg5 Rxc4 Qxf6 Kxf6 Rc1 Rcd4 b3 Rd3 Ke2 Rd2 Kf3 R7d3 Kg2 e6 Rhe1 Ke7 f5 Nd7 fxe6 fxe6 Kf1 Ne5 Re2 g5 Na4 Rd1 Re1 Rxe1 Kxe1 Rd7 Ke2 Nd3 Rc3 Nf4 Kf3 Kf6 Nc5 Rc7 h4 e5 hxg5 Kxg5 Rc4 b5 Rc1 b4 Rc4 a5 Na4 Rf7 Ke3 Ng2 Ke2 Nf4 Ke3 h5 Rc5 Ng2 Ke2 h4 Rxe5 Kg4 Re8 Nf4 Ke3 h3 f3 Kh4
Result: 0-1
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Na6 Bf4 c5 d5 e6 d6 e5 Bxe5 Nb4 Rd1 Be6 Qxc5 Nc2 Kd2 Nd7 Qc7 Nxe5 Qxd8 Raxd8 Kxc2 Ng4 Rd2 Bh6 Re2 Rxd6 h3 Nf6 g4 Nd7 e5 Rb6 Nd4 Nc5 b3 Rd8 Nxe6 Nxe6 Re4 Rc6 Bc4 a6 Rd1 Rdc8 Rd6 Nc5 Rxc6 Rxc6 Rd4 b5 b4 bxc4 bxc5 Rxc5 f4 Bf8 Rd7 Ra5 g5 h6 h4 hxg5 hxg5 Bb4 Nd5 Rxa2 Kb1 Rd2 e6 fxe6
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 b5 g4 h6 Qd2 Nbd7 O-O-O Bb7 h4 b4 Na4 d5 Bh3 g5 Bg2 gxh4 Rxh4 dxe4 g5 Nd5 Rxe4 hxg5 Bxg5 Qa5 f4 Rh2 Nxe6 fxe6 Rxe6 Kf7 Qd3 Bg7 Qf5 Kg8 Rxd5 Qxa4 Re7
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 c3 Nf6 Be2 Bd7 Qc2 Qc7 O-O g6 d4 cxd4 Nxd4 Nc6 Be3 Bg7 c4 Ng4 Bxg4 Bxg4 Qd2 O-O b3 Nxd4 Bxd4 Bxd4 Qxd4 Qc5 Qd3 a6 Nc3 Be6 a4 b6 Kh1 b5 axb5 axb5 Rxa8 Rxa8 Nxb5 Ra2 Kg1 Rb2 Nd4 Bd7 h3 Qa5 Rd1 Qa2 Qf1 Qa7 Qe1 f6 Qe3 Qa2 Qe1 Qa7 Kf1 Kf7 Rd2 Qa2 Rxb2 Qxb2 Qd1 h5 f3 g5 Ne2 g4 hxg4 hxg4 Kf2 Be6 Qd3 Bd7 f4 Bc6 Kg3 Qa1 Nc3 f5 Qe3 fxe4 Nxe4 Qf1 Ng5 Ke8 Qd2 Qb1 Kxg4 Qxb3 Qe2 Qb1 Kg3 Qf5 Nf3 Qg6 Kf2 Qf7 Kg3 Qg6 Ng5 Qf5 Qd2 e5 Nf3 Qxf4 Qxf4 exf4 Kxf4 Bb7 g3 Ba6 Nd2 Ke7 Ke4 Ke6 Kd4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 d5 c4 c6 Nc3 Nf6 Nf3 dxc4 a4 Bf5 Ne5 Nbd7 Nxc4 Qc7 g3 e5 dxe5 Nxe5 Bf4 Nfd7 Bg2 g5 Ne3 gxf4 Nxf5 O-O-O Qc2 Ng4 a5 fxg3 hxg3 a6 Ra4 Ndf6 Ne4 Nxe4 Bxe4 h5 Kf1 Kb8 Kg2 Be7 Nxe7 Qxe7 Bf3 Ne5 Bxh5 Qe6 Qc3 f6 Rah4 Qf5 Bf3 Rxh4 Rxh4 Qb1 Rh1 Rd1 Rxd1 Qxd1 b4 Kc7 Qc5 Qd6 Qxd6 Kxd6 Be4 Nc4 Bd3 Nb2 f4 Nd1 g4 Ne3 Kf3 Nd5 Ke4 Nxb4 Bc4 c5 g5 fxg5 fxg5 Ke7 Kf5 Nc2 Ke5 Ne3 Be6 c4
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nc3 d6 g3 Nc6 Bg2 g6 d3 Bg7 Be3 Nd4 Qd2 Qa5 f4 e6 Nf3 Ne7 O-O Nec6 e5 dxe5 Nxe5 O-O Rae1 f5 Bxc6 Nxc6 Nxc6 bxc6 Nd1 Qxa2 Bxc5 Rd8 b3 Rb8 Rf2 Qa6 Nc3 Rxb3 cxb3 Rxd3 Qa2 Rxc3 b4 Ra3 Qc2 Rd3 Rd2 Rd5 Rxd5 cxd5 Bf2 Bb7 Qc7 d4 Qd8 Bf8 Qf6 d3 Bc5 Bxc5 bxc5 d2 Qd8 Kg7 Qe7
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 cxd4 Nxd4 O-O Bg2 d5 cxd5 Nxd5 Qb3 Nc6 Nxc6 bxc6 O-O Qa5 Bd2 Bxc3 bxc3 Ba6 Rfd1 Qc5 e4 Bc4 Qa4 Nb6 Qb4 Qh5 Be3 Be2 Rd2 Rab8 Bxb6 axb6 Qd6 Bf3 Qxc6 Bxg2 Kxg2 Qe5 Qc4 Rfc8 Qd4 Qa5 Rb1 h6 Rb4 Qc5 Rd3 Qc7 a4 Rd8 Qe3 Rxd3 Qxd3 Rc8 Qb1 Qxc3 Rxb6 Rc4 Rb8 Kh7 Rb7 f6 Re7 Rb4 Qa2 Qc4 Qxc4 Rxc4 a5 Ra4 Rxe6 Rxa5 Rd6 Ra4 Kf3 Ra3 Ke2 h5 Rd3 Ra2 Ke3 Kg6 h3 Ra4 f4 Rb4 Ra3 Rc4 g4 hxg4 hxg4 Rb4 Ra6 Kf7 Ra7 Kg6 f5 Kh6 g5 fxg5 e5 g4 e6 Kg5
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Be3 c5 Qd2 cxd4 cxd4 Nc6 Rd1 Bg4 Be2 Bxe2 Nxe2 Na5 Qb4 O-O O-O b6 d5 Qd6 Qb5 Qe5 Rd4 Qb8 Ra4 Rc8 Nd4 Rc5 Qd7 Rc7 Qb5 Qc8 h3 Rc5 Qb1 Qd7 Rb4 Rac8 Nc6 Nxc6 Bxc5 Nxb4 Bxb4 Bc3 Rc1 Bxb4 Rxc8 Qxc8 Qxb4 Qc1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Nb6 Nc3 Qe6 Qe4 Bb4 Bd2 Ba6 b3 Bxc3 Bxc3 d5 Qh4 dxc4 Be2 Nd5 Bxc4 g5 Qd4 Bxc4 Qxc4 Nf4 Qxe6 Nxe6 O-O-O Ke7 Rhe1 Rhd8 Rxd8 Rxd8 Re4 Rd5 Kc2 c5 Ra4 Nd4 Bxd4 cxd4 Rxa7 Kd7 Kd3 Rxe5 Ra4 c5 b4 Ke6 Ra6 Kf5 b5 Rd5 Rc6 c4 Kxc4 d3 Kxd5 d2 g4
Result: 1-0
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nf3 Bg7 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Nc6 Nc3 Nxd4 Nxd4 Qxd4 Nb5 Qc4 a4 O-O b3 Qg4 Be3 Be6 Nxc7 Rad8 Qb1 Bxa1 Qxa1 Bxb3 a5 Nc4 Bh6 Qd4 Nd5 Qxa1 Nxe7 Kh8 Rxa1 Rfe8 Bg5 Rd6 Nd5 Rxe2 Bf6 Rxf6 Nxf6 Kg7 a6 bxa6 Nd7 a5 Nc5 a4 Nxa4 Re5 Nc3 Nd2 f4 Rc5 Ne2 a5 Kf2 Bc4 Ke3 Nb3 Ra3 Rc8 Nc3 Re8 Kf2 Rd8 Bf1 Be6 Be2 Rc8 Ne4 Bd5 Nd6 Rc6 Nb5 Rc1 Ke3 Re1 Kf2 Rc1 Ke3 Re1 Kf2 Ra1 Rxa1 Nxa1 Ke3 Nb3 Bd1 Kf6 Kd3 Nc5 Kd4 Bb3 Bf3 Nd7 Nd6 Ke6 Nb7 a4 Kc3 Nf6 Nc5 Kf5 Nxb3 axb3 Kxb3 Ng4 h3 Ne3 Be2 Ke4 Kc3 Nf5 Bc4 Nxg3 Bxf7 Ne2 Kd2 Nxf4 h4 Kf3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Bxf6 gxf6 Nd5 f5 c3 Bg7 exf5 Bxf5 Nc2 Be6 a4 O-O axb5 axb5 Rxa8 Qxa8 Nce3 Qb7 g4 Ne7 Bg2 Nxd5 Bxd5 Qe7 h4 Rc8 g5 Rc5 Be4 d5 Nxd5 Bxd5 Bxd5 e4 Ke2 Qe5 Bb3 Qc7 Qg1 b4 g6 hxg6 Qxg6 Kf8 Qg3 Qxg3 fxg3 bxc3 bxc3 Rxc3 Bd5 Rxg3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 h3 O-O Be2 b5 Be3 Bb7 Qd3 cxd4 cxd4 Nd7 O-O Nb6 Qb1 Na4 Qxb5 Nc3 Qxb7 Nxe2 Kh1 Nxd4 Rad1 e5 Nxe5 Bxe5 f4 Bg7 e5 Qb6 Qxb6 axb6 Bxd4 Rxa2 Bxb6 Re8 Bd8 Ra6 Bc7 Rae6 g4 g5 f5 Rc6 Rd7 Bxe5 Re1 f6 Bxe5 Rxe5 Rxe5 fxe5 Re7 h5 Kg2 hxg4 hxg4 Rc4 Kf3 Rf4 Kg3 e4
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 h3 Nf6 Bc4 Qb6 O-O O-O Nde2 Qxb2 Bb3 Qa3 f4 Nc6 Kh1 Be6 Qd3 Rac8 fxg5 hxg5 Nd5 Rfe8 Rad1 Nb4 Qf3 Nbxd5 exd5 Bd7 c3 a5 Qd3 a4 Bc2 Qc5 Rxf6 exf6 Qh7 Kf8 Nd4 Re5 Bxe5 fxe5 Ne6 Bxe6 dxe6 Rc7 Bxa4 d5 Qf5 Qc4 Bd7 Qf4 Qb1 fxe6 Bxe6 Ke7 Bxd5 Rd7 c4 Qe3 Qh7 Kd8 Rb1 Qf4 Be6 Re7 Bg4 Rf7 Qd3 Qd4 Qg6
Result: 1-0
Processing game...
FEN: 
Moves: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 Qc2 Be7 e3 Nbd7 Bd3 O-O Nge2 Re8 O-O Nf8 f3 Ng6 Rad1 h6 Bxf6 Bxf6 Bxg6 fxg6 e4 g5 e5 Be7 f4 gxf4 Nxf4 Rf8 Ng6 Rxf1 Rxf1 Be6 Ne2 Qd7 h4 Re8 Ng3 Bf7 Nxe7 Rxe7 Nf5 Re6 Nd6 Bg6 Qc3 Rxd6 exd6 Qxd6 Qa3 Qb8 Qe7 Qg3 Rf3 Qg4 Qf8 Kh7 Qf4 Qd7 Rg3 Be4 Qe5 Qf7 h5 Qd7 b4 a6 a4 Kg8 a5 Kh7 Kf2 Kg8 Qb8 Kh7 Qf8 Bc2 Qf4 Be4 Qe5 Kg8 Ke3 Kh7 Ke2 Kg8 Qb8 Kh7 Qf8 Bc2 Kd2 Be4 Kc1 Qc7 Rg4 Qd7 Rf4 Bxg2 Qf5 Qxf5 Rxf5 Kg8 Kd2 Bh3 Rf4 Be6 Ke3 Bf7 Rf5 Kf8 Re5 Be8 Kf4 Bf7 Kg4 Be8 Kg3 Bf7 Kf4 Be8 Kf5 Bxh5 Ke6 Bg6 Re3 Bc2 Rf3 Kg8 b5 cxb5 Kxd5 b4 Kc4 b3 Kc3 g5 d5 g4 Rf4 h5 d6
Result: 1-0
Processing game...
FEN: 
Moves: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 b5 Bd3 Qb6 Nb3 Qc7 O-O Bb7 a3 Nf6 f4 d6 Kh1 Nbd7 Qe2 Be7 Nd4 O-O Bd2 Nc5 Rae1 Nfd7 Be3 Bf6 Bg1 Rfe8 Qf3 g6 Qh3 Bg7 Nf3 f5 Ng5 Nf8 exf5 exf5 Qg3 h6 Nf3 Nxd3 cxd3 Qf7 Nd4 Rxe1 Rxe1 Re8 h3 Ne6 Nc2 Kh7 Qe3 g5 fxg5 hxg5 Bh2 f4 Qf2 Nc5 Rxe8 Qxe8 Ne1 Bd4 Qxd4 Qxe1 Bg1 Qd2 Ne4 Bxe4 dxe4 Qxd4 Bxd4 Nxe4 Kg1 Kg6 Kf1 Kf5 Ke2 Nc5 Kf3 Nd3 Ke2 Ne5 Bb6 Nc4 Bd4 a5 b3 Nxa3 Kd3 a4 Bb2 Nb1 Kc2 a3 Bg7 a2 Kb2 Nd2 Kxa2 Nf1 Ka3 Ne3 Kb4 Nxg2 Bd4 Ne3
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 b5 g4 h6 Qd2 Nbd7 O-O-O Bb7 h4 b4 Na4 Qa5 b3 Be7 Kb1 Nc5 Nxc5 dxc5 Ne2 Rd8 Qc1 Nd7 Ng3 Ne5 Be2 Rxd1 Qxd1 Qc7 Bf4 Bd6 Nh5 g6 Nf6 Ke7 g5 h5 Be3 Nc6 f4 Rd8 Qf1 Nd4 Bd3 Kf8 Rg1 Kg7 Bxd4 cxd4 e5 Be7 Qe2 a5 f5 exf5 Bxf5 d3 Bxd3 Bc5 e6 Bxg1 e7 Bc6 Qe5 Qxe5 exd8=Q Qe1 Kb2 Qc3 Kb1 Qe1 Kb2 Qc3 Kb1 Qe1
Result: 1/2-1/2
Processing game...
FEN: 
Moves: d4 Nf6 Nf3 d5 c4 e6 Nc3 c6 e3 Nbd7 Qc2 Bd6 Bd3 O-O O-O dxc4 Bxc4 a6 a4 c5 Rd1 Qc7 Ne4 Nxe4 Qxe4 Nf6 Qh4 cxd4 Bd3 h6 e4 Nd7 Bxh6 gxh6 Rac1 Qd8 Qxh6 Qf6 Qxf6 Nxf6 e5 Be7 exf6 Bxf6 Be4 Rd8 g4 Rb8 g5 Bh8 Ne1 Bd7 a5 Ba4 Rd2 d3 Rxd3 Bb5 Rxd8 Rxd8 Nf3 Bxb2 Rc7 Ba3 Rxb7 Rd1 Kg2 Bd6 Kh3 Rc1 Nd4 Bf1 Bg2 Rc3 f3 Bd3 Rb3 Rxb3 Nxb3 Bb4 f4 Bc4 Nd4 Bxa5 f5 Bc3 Nc6 a5 Nxa5 Bxa5 Be4 Bd5 Bd3 Bc7 f6 Bf4 Kg4 Bxh2 g6 e5 Kf5 Be6 Kg5 Bf4 Kh5 Bd5 Kg4 fxg6 Bxg6 Be6 Kf3 Bg5
Result: 0-1
Processing game...
FEN: 
Moves: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Nbd7 Qd2 b5 g4 Nb6 g5 Nfd7 Nd5 Rc8 Nxb6 Nxb6 Qa5 Nc4 Bxc4 bxc4 Qxd8 Kxd8 Na5 h6 gxh6 g6 O-O-O Kc7 b3 Bxh6 Bxh6 Rxh6 bxc4 g5 Rhg1 Kb6 Nb3 Bxc4 Rd2 Rg6 Kb2 Kc7 Rdg2 Rh6 Nd2 Be6 Rxg5 Rxh2 R5g2 Rh3 c4 Kd7 Kc3 a5 Rg3 Rhh8 Rc1 Rc7 Rgg1 Rhc8 Kd3 Ke7 Rc3 d5 exd5 Bxd5 Re1 f6 f4 Kd6 fxe5 fxe5 Rec1 Bc6 Kc2 Rg7 Rd3 Kc7 Re3 Rd8 Kc3 Rg2 Rc2 a4 Nb1 Rxc2 Kxc2 Kd6 Rd3 Ke7 Rxd8 Kxd8 Kc3 Kd7 Kb4 Kd6 Nc3 e4 Nd1 Ke5 Kc5 Be8 a3 Kf4 Kd4 Kf3 Ne3 Bc6 c5 Kf4 Nc4 Bb5 Nb6 Bc6 Nc4 Bb5 Ne3
Result: 1/2-1/2
Processing game...
FEN: 
Moves: e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Nb6 Nc3 Qe6 Qe4 Ba6 b3 O-O-O Bb2 Bb7 O-O-O Re8 f4 d5 cxd5 cxd5 Qc2 Kb8 Kb1 g6 Be2 c5 Bf3 Be7 g4 d4 Bxb7 Kxb7 Ne4 Rc8 Qg2 Kb8 Rhe1 h5 g5 Rhd8 Qf3 Qf5 Ka1 Nd5 Nd6 Rxd6 exd6 Bxd6 Re4 Qg4 Qf1 Bc7 Rc1 Bb6 Re5 Ne3 Qh1 Qxf4 Re7 Rc7 Re8 Rc8 Rxc8 Kxc8 Qc6 Kd8 a4 Qc7 Qf6 Qe7 Bxd4 Qxf6 Bxf6 Kd7 h3 Nd5 Rd1 Ke6 Re1 Kd7 Be5 Ba5 Re2 Bd8 h4 Ba5 Kb2 Ne7 Bg3 Nf5 Bf2 Kc6 Re5 Bb4 Kc2 Kd6 Re8 a5 Kd3 Kd7 Ra8 Ke6 Ra6 Kd5 Rf6 Nd6 Be3 c4 bxc4 Nxc4 Bf4 Nb2 Kc2 Nxa4 Rxf7 Nc5 Rf6 Be1 Rxg6 Bxh4 Rd6 Ke4 g6 Kxf4 g7 Ne4 Rc6 Nf6 Rxf6 Bxf6 g8=Q h4 Kd3 Bg5 Qf7 Kg3 Ke4 h3 Qf3 Kh2 Qf2 Kh1 Qg3
Result: 1-0
An error occurred during parsing: Null check operator used on a null value

""";

List<String> gameTokens(List<String> moveList) {
  // final Game gameToPlay = Game();
  // int moveCount = 0;
  List<String> parsedMoves = [];
  for (String moveToken in moveList) {
    // Limit to 50 moves to prevent infinite loops
    // final predictedMove = await model.predictNextMove(gameToPlay);
    // print('Model predicted: $predictedMove');

    // Check if the predicted move is legal in the current position
    // final legalMoves = gameToPlay.generateLegalMoves();
    final move = gameToPlay.getMoveSan(moveToken, checks: false);

    if (move != null) {
      parsedMoves.add(gameToPlay.toAlgebraic(move));
      // final move = gameToPlay.getMove(predictedMove)!;
      gameToPlay.makeMove(move);
      // print('Legal move: $move');
    } else {
      // print('Predicted an illegal move. Terminating game.');
      break;
    }
  }

  return parsedMoves;
}

// class Move {
//   int from;
//   int to;
//   // String? prom;
//   // int? promSq;
//   String? promPiece;

//   static int rank(int index) {
//     return (index / 8).toInt();
//   }

//   static int file(int index) {
//     return (index % 8).toInt();
//   }

//   int toIndex(int row, int col) {
//     // 1. Safety Check: Prevent RangeErrors before they happen
//     if (rows > this.shape[0] || cols > this.shape[1]) {
//       throw RangeError(
//         "Slice dimensions [$rows, $cols] exceed Tensor shape ${this.shape}",
//       );
//     }

//     final out = Tensor([rows, cols], children: {this});
//     final int stride = this.shape[1]; // The original width

//     // 2. Forward Pass
//     for (int i = 0; i < rows; i++) {
//       for (int j = 0; j < cols; j++) {
//         out.data[i * cols + j] = this.data[i * stride + j];
//       }
//     }

//     // 3. Backward Pass: Map gradients from the small slice back to the big tensor
//     out.onBackward = () {
//       for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//           // We use += because multiple slices might overlap or be used multiple times
//           this.grad[i * stride + j] += out.grad[i * cols + j];
//         }
//       }
//     };

//     return out;
//   }

//   Move({required this.from, required this.to, this.promPiece});
//   factory Move.fromString(String move) {
//     if (move.length == 4) {
//       return Move(from: from, to: to);
//     }
//   }

//   static const strToInt = {
//     'a': 0,
//     'b': 1,
//     'c': 2,
//     'd': 3,
//     'e': 4,
//     'f': 5,
//     'g': 6,
//     'h': 7,
//   };

//   static const intToRank = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7};
// }

List<List<String>> dataset(int size) {
  List<String> parts = pgn.split("Moves: ");
  parts = List.generate(parts.length, (index) {
    String part = parts[index];
    final lastPart = part.indexOf("\nResult");
    if (lastPart != -1) {
      return part.substring(0, lastPart);
    }
    return "";
  });

  List<List<String>> gameLines = [];
  int parsed = 0;
  for (String gameline in parts) {
    parsed++;
    print("Parsing game: $parsed");
    final moveList = gameTokens(gameline.split(" "));
    gameLines.add(moveList);
    if (parsed > size) break;
  }

  // print(gameLines.where((test) => test.isNotEmpty));
  return gameLines.where((test) => test.isNotEmpty).toList();
}

void main() {
  print(dataset(5));
}
