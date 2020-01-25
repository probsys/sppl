## Query Language For A Sum-Product Probabilistic DSL

### Installation and Tests

Please install Python 3 dependencies in `requirements.txt`.

Run the following command in the shell:

    $ ./check.sh

### Overview of the domain specific language

Consider the following probabilistic DSKL, specified in Haskell notation:

    type VarName = String

    data Distribution
      = Normal Double Double
      | Gamma Double Double
      | Cauchy Double
      | Poisson Double
      | Binomial Integer Double

    data Invertible
      = Identity
      | Abs Invertible
      | Exp Double Invertible
      | Log Double Invertible
      | Pow Double Invertible
      | Poly [Double] Invertible

    data Network
      = Primitive VarName Invertible Distribution
      | Sum [Network] [Double]
      | Product [Network]
      | Condition Network Event

    data Interval
      = CC Double Double -- Closed Closed
      | CO Double Double -- Closed Open
      | OC Double Double -- Open Closed
      | OO Double Double -- Open Open

    data Event
      = Between Interval Invertible VarName
      | Contains [Integer] VarName
      | Or Event Event
      | And Event
      | Not Event

    data SampleExpr = Invertible VarName | Event

A probabilistic program is any expression of type `Network` in this DSL.
Its semantics are given by the usual sum-product network.

  - Internal nodes are `Sum` `Product` expressions.

  - Leaf nodes are either a `Primitive` distribution named `x`, or an
    `Transform` (of type `Invertible`) of a `Primitive` distribution.

The higher-order constructor `Condition` takes in an arbitrary `Network`
and a probabilistic `Event` and returns a new `Network` representing the
conditional distribution given the event.

### Finding the probability of and simulating an event

Given a probabilistic program `network`, the key query is finding the log
probability of a given `event`:

    logprob:: Dist -> Event -> Real

The conditional probability of an event is obtained by querying a conditioned
network, for example

    logprob
      (Condition
          -- Network
         (Sum
            [ (Primitive "X"
                (Poly [1.2, 1.1, -7] (Log Identity)) $ Normal 0 1)
            , (Primitive "X" Identity $ Gamma 0 1)
            ]
            [0.7, 0.3])
          -- Conditioning event
         (Between (ClCl 0 10) Identity "X") )
      -- Query event
      (Or (Contains [10, 12, 14] "X") (Between (ClOp (-10) 12) (Log Identity) "X"))

If the cumulative probabilities of the `Primitive` distributions (on either
finite, countable, or uncountable domains) are known then exact inference in the
network is possible using symbolic analysis with fixed runtime.

To simulate an event from a (possibly conditioned) `network` use:

    simulate :: Dist -> [SampleExpr] -> [Real]

TODO: Add example

### Finding the mutual information between events

Given a `network`, the mutual information of `eventA` and `eventB`
given `eventC` is computed by the following expression:

    let network' = condition network eventC
    in let lpA1 = logprob network' eventA
    in let lpB1 = logprob network' eventB
    in let lpA0 = 1 - lpA1
    in let lpB0 = 1 - lpB1
    in let lp00 = logprob network' (And (Not eventA) (Not eventB))
    in let lp01 = logprob network' (And (Not eventA) eventB)
    in let lp10 = logprob network' (And (eventA (Not eventB)))
    in let lp11 = logprob network' (And (eventA eventB))
    in let m00 = (exp lp00) * (lp00 - (lpA0 + lpB0))
    in let m01 = (exp lp01) * (lp01 - (lpA0 + lpB1))
    in let m10 = (exp lp10) * (lp10 - (lpA1 + lpB0))
    in let m11 = (exp lp11) * (lp11 - (lpA1 + lpB1))
    in m00 + m01 + m10 + m11
