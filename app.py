import math
import os
from typing import Callable, Dict

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


def generate_terminal_prices(
    s0: float, r: float, sigma: float, maturity: float, n_paths: int = 20000
) -> np.ndarray:
    """Simulate terminal asset prices under risk-neutral GBM."""
    z = np.random.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * maturity
    diffusion = sigma * math.sqrt(maturity) * z
    return s0 * np.exp(drift + diffusion)


def build_payoff_function(expression: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a vectorized payoff function from a user-provided expression.

    The expression can reference the terminal price as `S`. A small set of
    NumPy helpers is exposed for convenience.
    """

    allowed_names: Dict[str, object] = {
        "np": np,
        "numpy": np,
        "S": None,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "maximum": np.maximum,
        "minimum": np.minimum,
    }

    code = compile(expression, "<user_payoff>", "eval")

    def payoff(s: np.ndarray) -> np.ndarray:
        local_env = {**allowed_names, "S": s}
        return np.array(eval(code, {"__builtins__": {}}, local_env))

    return payoff


def price_option(s0: float, r: float, sigma: float, maturity: float, expression: str) -> Dict[str, float]:
    terminal_prices = generate_terminal_prices(s0, r, sigma, maturity)
    payoff_fn = build_payoff_function(expression)
    payoffs = payoff_fn(terminal_prices)
    discounted = np.exp(-r * maturity) * payoffs
    return {
        "price": float(np.mean(discounted)),
        "std_error": float(np.std(discounted, ddof=1) / math.sqrt(len(discounted))),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_defaults = {
        "s0": "100",
        "r": "0.05",
        "sigma": "0.2",
        "maturity": "1",
        "payoff": "maximum(S - 100, 0)",
    }

    if request.method == "POST":
        try:
            s0 = float(request.form.get("s0", form_defaults["s0"]))
            r = float(request.form.get("r", form_defaults["r"]))
            sigma = float(request.form.get("sigma", form_defaults["sigma"]))
            maturity = float(request.form.get("maturity", form_defaults["maturity"]))
            expression = request.form.get("payoff", form_defaults["payoff"]).strip()

            result = price_option(s0, r, sigma, maturity, expression)
            result.update({
                "s0": s0,
                "r": r,
                "sigma": sigma,
                "maturity": maturity,
                "expression": expression,
            })
        except Exception as exc:  # pylint: disable=broad-except
            error = f"Error during calculation: {exc}"

    return render_template("index.html", result=result, error=error, defaults=form_defaults)


def main():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
