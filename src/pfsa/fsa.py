import json
import pickle
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

import numpy as np
from tqdm import tqdm, trange  # type: ignore

String = Union[str, Sequence[Union[str, int]]]


def H(p: np.ndarray) -> float:
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class PFSA:
    def __init__(
        self,
        n_states: Optional[int] = None,
        n_symbols: Optional[int] = None,
        fname: Optional[str] = None,
    ):
        assert n_states is not None and n_symbols is not None or fname is not None

        if n_states is not None and n_symbols is not None:
            self.n_states = n_states
            self.n_symbols = n_symbols
            self.λ = np.zeros(n_states)
            self.Ts: Dict[int, np.ndarray] = {
                y: np.zeros((n_states, n_states)) for y in range(n_symbols)
            }
            self.ρ = np.zeros(n_states)
            self.q2str = {i: str(i) for i in range(n_states)}
            self.q_struct: Dict[int, Tuple[int, ...]] = {
                i: (i,) for i in range(n_states)
            }

        elif fname is not None:
            self.load(fname)

        self.theme = "dark"

    def bos_model(self, b: int) -> "PFSA":
        N, M = self.n_states, self.n_symbols
        BOS = M
        B = PFSA(N + b, self.n_symbols + 1)
        B.λ = np.zeros(B.n_states)
        B.λ[N] = 1
        B.ρ[:N] = self.ρ
        B.Ts = {y: np.zeros((B.n_states, B.n_states)) for y in range(B.n_symbols)}
        for q in self.Q:
            for y, t, w in self.arcs(q):
                B.Ts[y][q, t] = w
        for q in range(N, N + b - 1):
            B.Ts[BOS][q, q + 1] = 1
        B.Ts[BOS][N + b - 1, 0] = 1
        return B

    def arcs(self, q: int):
        for y in range(self.n_symbols):
            for t in range(self.n_states):
                if self.Ts[y][q, t] > 0:
                    yield y, t, self.Ts[y][q, t]

    def next_symbol_probabilities(self, q: int) -> np.ndarray:
        return np.asarray(
            [self.Ts[y][q].sum() for y in range(self.n_symbols)] + [self.ρ[q]]
        )

    @property
    def next_symbol_probability_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.Ts[y][q].sum() for y in range(self.n_symbols)] + [self.ρ[q]]
                for q in range(self.n_states)
            ]
        ).T

    def next_state_probabilities(self, q: int, y: Optional[int] = None) -> np.ndarray:
        if y is None:
            return self.T[q] / self.T[q].sum()
        return self.Ts[y][q] / self.Ts[y][q].sum()

    @property
    def Q(self):
        return set(range(self.n_states))

    @property
    def I(self):  # noqa: E741, E743
        return set(q for q, w in enumerate(self.λ) if w > 0)

    @property
    def F(self):
        return set(q for q, w in enumerate(self.ρ) if w > 0)

    @property
    def T(self):
        return sum(self.Ts.values())

    def Ty(self, y: String) -> np.ndarray:
        Ty = np.eye(self.n_states)
        for s in y:
            Ty = Ty @ self.Ts[int(s)]
        return Ty

    @property
    def α(self) -> np.ndarray:
        return self.λ @ self.kleene

    @property
    def β(self) -> np.ndarray:
        return self.kleene @ self.ρ

    @property
    def kleene(self) -> np.ndarray:
        return np.linalg.inv(np.eye(self.n_states) - self.T)

    @property
    def ξ(self) -> np.ndarray:
        ξ = np.zeros(self.n_states)
        for q in self.Q:
            h_ρ = self.ρ[q] * np.log2(self.ρ[q]) if self.ρ[q] > 0 else 0
            outgoing_weights = [
                self.Ts[y][q].sum()
                for y in range(self.n_symbols)
                if self.Ts[y][q].sum() > 0
            ]
            ξ[q] = -sum(w * np.log2(w) for w in outgoing_weights) - h_ρ

        return ξ

    @property
    def allsum(self) -> float:
        return self.λ.T @ self.β

    @property
    def entropy(self) -> float:
        return self.α @ self.ξ

    @property
    def next_symbol_entropy(self) -> float:
        p = self.normalized_limit_state_distribution
        H_ = 0.0
        for q in self.Q:
            H_ += p[q] * H(self.next_symbol_probabilities(q))

        return H_
        # Equivalent:
        # return self.entropy / (self.mean_length + 1)
        # return (sum(self.α[q] * H(self.next_symbol_probabilities(q))))
        # / (self.mean_length - 1)

    def length_p(self, t: int) -> float:
        """Computes the mass of length-t strings.

        Args:
            t (int): The length t.

        Returns:
            float: The mass of length-t strings.
        """
        from numpy.linalg import matrix_power
        return self.λ @ matrix_power(self.T, t) @ self.ρ
        # NOT: self.λ @ self.T**t @ self.ρ

    @property
    def mean_length(self) -> float:
        return self.α @ np.ones(self.n_states) - 1  # type: ignore
        # Equivalent:
        # return self.α @ self.T @ self.kleene @ self.ρ

    def local_entropy(self, m: int) -> float:
        """Computes the m-local entropy of the PFSA.

        Args:
            m (int): The order m of the entropy.

        Returns:
            float: The m-local entropy of the PFSA.
        """

        Hs = []
        infix_probabilities = []

        for c in tqdm(
            product(range(self.n_symbols), repeat=m - 1),
            total=self.n_symbols ** (m - 1),
            desc=f"Computing {m}-local entropy",
        ):
            # for c in product(range(self.n_symbols), repeat=m - 1):
            infix_probability = self.infix_probability(c)
            if infix_probability < 1e-6:
                continue
            Hs.append(infix_probability * H(self.infix_next_symbol_probabilities(c)))
            infix_probabilities.append(infix_probability)

        return sum(Hs) / sum(infix_probabilities)


    @property
    def limit_state_distribution(self) -> np.ndarray:
        """Computes the limit state distribution of the PFSA.

        Returns:
            np.ndarray: The limit state distribution of the PFSA.
        """
        return self.α

    @property
    def normalized_limit_state_distribution(self) -> np.ndarray:
        """Computes the limit state distribution of the PFSA.

        Returns:
            np.ndarray: The limit state distribution of the PFSA.
        """
        p = self.limit_state_distribution
        return p / p.sum()

    def state_distribution(self, y: String) -> np.ndarray:
        """Computes the distribution over states after observing the prefix y.

        Args:
            y (String): The prefix y.

        Returns:
            float: The distribution over states after observing the prefix y.
        """
        return self.λ @ self.Ty(y)

    def infix_state_distribution(self, c: String) -> np.ndarray:
        """Computes the distribution over states after observing c preceeded
        by any prefix.

        Args:
            c (String): The context c.

        Returns:
            float: The distribution over states after observing the prefix c preceeded
            by any prefix.
        """
        p = self.α @ self.Ty(c)
        return p / p.sum()

    def prefix_probability(self, y: String) -> float:
        """Computes the prefix probability of y.

        Args:
            y (String): The prefix y.

        Returns:
            float: The prefix probability of y.
        """
        # NOTE: This is the same as
        # return self.state_distribution(y) @ self.kleene @ self.ρ
        return self.state_distribution(y) @ np.ones(self.n_states)  # type: ignore

    @property
    def prefix_probability_normalizer(self) -> float:
        """Computes the prefix probability normalizer.

        Returns:
            float: The prefix probability normalizer.
        """
        return self.α @ np.ones(self.n_states)  # type: ignore

    def normalized_prefix_probability(self, y: String) -> float:
        """Computes the normalized prefix probability of y.

        Args:
            y (String): The prefix y.

        Returns:
            float: The normalized prefix probability of y.
        """
        return self.prefix_probability(y) / self.prefix_probability_normalizer

    def infix_probability(self, y: String) -> float:
        """Computes the infix probability of y.

        Args:
            y (String): The infix y.

        Returns:
            float: The infix probability of y.
        """
        # NOTE: This is the same as
        # return self.α @ self.Ty(y) @ self.kleene @ self.ρ
        return self.α @ self.Ty(y) @ np.ones(self.n_states)

    def infix_probability_normalizer(self, m: int = 0) -> float:
        """Computes the infix probability normalizer.

        Returns:
            float: The infix probability normalizer.
        """
        if m > 0:
            return sum(
                self.infix_probability(c)
                for c in product(range(self.n_symbols), repeat=m - 1)
            )
        else:
            return self.α @ self.kleene @ np.ones(self.n_states)

    def normalized_infix_probability(self, y: String, m: int = 0) -> float:
        """Computes the normalized infix probability of y.

        Args:
            y (String): The infix y.

        Returns:
            float: The normalized infix probability of y.
        """
        return self.infix_probability(y) / self.infix_probability_normalizer(m)

    def infix_next_symbol_probabilities(self, c: String) -> np.ndarray:
        """Computes the distribution over next symbols after observing the infix c.

        Args:
            c (String): The infix c.

        Returns:
            float: The distribution over next symbols after observing the infix c.
        """
        return self.next_symbol_probability_matrix @ self.infix_state_distribution(c)

    def prefix_next_symbol_probabilities(self, c: String) -> np.ndarray:
        """Computes the distribution over next symbols after observing the prefix c.

        Args:
            c (String): The prefix c.

        Returns:
            float: The distribution over next symbols after observing the prefix c.
        """
        return self.next_symbol_probability_matrix @ self.state_distribution(c)

    def suffix_probability(self, y: String) -> float:
        """Computes the suffix probability of y.

        Args:
            y (String): The suffix y.

        Returns:
            float: The suffix probability of y.
        """
        return self.α @ self.Ty(y) @ self.ρ

    def probability(self, y: String) -> float:
        """Computes the probability of y.

        Args:
            y (String): The string y.

        Returns:
            float: The probability of y.
        """
        return np.dot(self.state_distribution(y), self.ρ)

    @property
    def is_probabilistic(self) -> bool:
        """Checks if the PFSA is probabilistic.

        Returns:
            bool: True if the PFSA is probabilistic, False otherwise.
        """
        for q in range(self.n_states):
            if not np.isclose(
                sum(self.Ts[y][q].sum() for y in range(self.n_symbols)) + self.ρ[q],
                1,
            ):
                return False
        return True

    def intersect(self, A: "PFSA") -> "PFSA":
        """Intersects the PFSA with another PFSA.

        Args:
            A (PFSA): The PFSA to intersect with.

        Returns:
            PFSA: The PFSA resulting from the intersection.
        """

        B = PFSA(self.n_states * A.n_states, self.n_symbols)

        def q2(p, q):
            B.q2str[p * A.n_states + q] = f"({self.q2str[p]}, {A.q2str[q]})"
            B.q_struct[p * A.n_states + q] = (self.q_struct[p], A.q_struct[q])
            return p * A.n_states + q

        for p, q in product(self.I, A.I):
            B.λ[q2(p, q)] = self.λ[p] * A.λ[q]

        for p, q in product(self.F, A.F):
            B.ρ[q2(p, q)] = self.ρ[p] * A.ρ[q]

        for p, q in product(self.Q, A.Q):
            for y in range(self.n_symbols):
                for s, t in product(range(self.n_states), range(A.n_states)):
                    B.Ts[y][q2(p, q), q2(s, t)] = self.Ts[y][p, s] * A.Ts[y][q, t]

        return B.trim()

    @property
    def accessible(self) -> Set[int]:
        """Computes the set of accessible states.

        Returns:
            Set[int]: The set of accessible states.
        """
        accessible = set()
        stack = list(self.I)
        while stack:
            q = stack.pop()
            accessible.add(q)
            for _, t, _ in self.arcs(q):
                if t not in accessible:
                    stack.append(t)
        return accessible

    @property
    def coaccessible(self) -> Set[int]:
        """Computes the set of coaccessible states.

        Returns:
            Set[int]: The set of coaccessible states.
        """
        return self.reverse().accessible

    def reverse(self) -> "PFSA":
        """Computes the reverse of the PFSA.

        Returns:
            PFSA: The reverse of the PFSA.
        """
        B = PFSA(self.n_states, self.n_symbols)
        B.λ = self.ρ
        B.ρ = self.λ
        B.Ts = {
            y: np.zeros((self.n_states, self.n_states)) for y in range(self.n_symbols)
        }
        for q in self.Q:
            for y, t, w in self.arcs(q):
                B.Ts[y][t, q] = w
        return B

    @property
    def useful(self) -> Set[int]:
        """Computes the set of useful states.

        Returns:
            Set[int]: The set of useful states.
        """
        return self.accessible & self.coaccessible

    def trim(self) -> "PFSA":
        """Trims the PFSA.

        Returns:
            PFSA: The trimmed PFSA.
        """
        useful = self.useful

        def q2(q):
            return {q: i for i, q in enumerate(sorted(useful))}[q]

        B = PFSA(len(useful), self.n_symbols)
        B.q2str = {q2(q): self.q2str[q] for q in useful}
        B.q_struct = {q2(q): self.q_struct[q] for q in useful}
        B.λ = np.asarray([self.λ[q] for q in useful])
        B.ρ = np.asarray([self.ρ[q] for q in useful])
        B.Ts = {y: np.zeros((B.n_states, B.n_states)) for y in range(self.n_symbols)}
        for q in self.Q:
            if q not in useful:
                continue
            for y, t, w in self.arcs(q):
                if t in useful:
                    B.Ts[y][q2(q), q2(t)] = w
        return B

    def sample(
        self, n: int, logp: bool = False, to_string: bool = False
    ) -> List[Union[String, Tuple[String, float]]]:
        return [self.sample_one(logp, to_string) for _ in trange(n)]

    def sample_one(
        self, logp: bool, to_string: bool
    ) -> Union[String, Tuple[String, float]]:
        q = np.random.choice(list(self.Q), p=self.λ)
        y, logp_ = [], 0
        while True:
            p = self.next_symbol_probabilities(q)
            a = np.random.choice(self.n_symbols + 1, p=p)
            y.append(a)
            if a == self.n_symbols:
                logp_ += np.log(self.ρ[q])
                break
            logp_ += np.log(p[a])
            q = np.random.choice(self.n_states, p=self.next_state_probabilities(q, a))

        if logp:
            return (" ".join(map(str, y)), logp_) if to_string else (y, logp_)
        return " ".join(map(str, y)) if to_string else y

    def save(self, filename: str):
        data = {
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "λ": self.λ,
            "ρ": self.ρ,
            "Ts": self.Ts,
            "q2str": self.q2str,
            "q_struct": self.q_struct,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.n_states = data["n_states"]
        self.n_symbols = data["n_symbols"]
        self.λ = data["λ"]
        self.ρ = data["ρ"]
        self.Ts = data["Ts"]
        self.q2str = data["q2str"]
        self.q_struct = data["q_struct"]

    def __call__(self, y: String) -> float:
        return self.probability(y)

    def __mul__(self, A: "PFSA") -> "PFSA":
        return self.intersect(A)

    def __repr__(self):
        r = f"PFSA(n_states={self.n_states}, n_symbols={self.n_symbols})\n"
        for y in range(self.n_symbols):
            r += f"\nT[{y}]: \n{self.Ts[y]}\n"
        r += f"\nT: \n{self.T}\n"
        r += f"\nT*: \n{self.kleene}\n"
        r += f"\nλ = {self.λ}\n"
        r += f"\nρ = {self.ρ}\n"
        return r

    def ascii_visualize(self):
        """
        ASCII visualization of the FST
        """
        ret = []
        for q in self.I:
            if q in self.F:
                label = f"{str(q)} / [{self.λ[q]:.3f} / {self.ρ[q]:.3f}]"
            else:
                label = f"{str(q)} / {self.λ[q]:.3f}"
            ret.append(f"({label})")

        for q in (self.Q - self.F) - self.I:
            ret.append(f"({str(q)})")

        for q in self.F:
            if q in self.I:
                continue
            ret.append(f"({str(q)} / {self.ρ[q]:.3f})")

        for q in self.Q:
            for y, t, w in self.arcs(q):
                ret.append(f"{str(q)} --{y} / {w:.3f}--> {str(t)}")

        return "\n".join(ret)

    def _repr_html_(self):  # noqa: C901
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """

        def w2s(w):
            return f"{w:.3f}"

        ret = []
        if self.n_states == 0:
            return "<code>Empty FST</code>"

        if self.n_states > 64:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(n_states={self.n_states})</code>"
            )

        # print initial
        for q in self.I:
            if q in self.F:
                label = f"{self.q2str[q]} / [{w2s(self.λ[q])} / {w2s(self.ρ[q])}]"
                color = "af8dc3"
            else:
                label = f"{self.q2str[q]} / {w2s(self.λ[q])}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{q}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )

            ret.append(f'g.node("{q}").style = "fill: #{color}"; \n')

        # print normal
        for q in (self.Q - self.F) - self.I:
            lbl = self.q2str[q]

            ret.append(
                f'g.setNode("{q}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{q}").style = "fill: #8da0cb"; \n')

        # print final
        for q in self.F:
            # already added
            if q in self.I:
                continue

            lbl = f"{self.q2str[q]} / {w2s(self.ρ[q])}"

            ret.append(
                f'g.setNode("{q}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{q}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for y, t, w in self.arcs(q):
                label = f"{y} / {w2s(w)}"
                to[t].append(label)

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label, qrep, drep = json.dumps("\n".join(values)), q, repr(d)
                color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
                edge_string = (
                    f'g.setEdge("{qrep}","{drep}",{{arrowhead:"vee",'
                    + f'label:{label},"style": "stroke: {color}; fill: none;", '
                    + f'"labelStyle": "fill: {color}; stroke: {color}; ", '
                    + f'"arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
                )
                ret.append(edge_string)

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(n_states={self.n_states})</code>"
            )

        ret2 = [
            """
       <script>
       try {
       require.config({
       paths: {
       "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
       "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
       }
       });
       } catch {
       ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
       "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
            function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            }
        )
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        """
        ]

        obj = "fst_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            """
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width")-g.graph().width*initialScale)/2,20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();

        </script>
        """
        )

        return "".join(ret2)
