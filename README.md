## Query Language For A Sum-Product Probabilistic DSL

### Installation and Tests

Please install the python dependencies in `requirements.txt`.

Run the following command in the shell:

    $ ./check.sh

### Overview

Consider the following probabilistic domain-specific language:

    Dist = (Primitive symbol theta)
            | (Transform (Primitive symbol theta) func)
            | (Mixture Dist... weights)
            | (Product Dist...)
            | (Condition Dist event)

A probabilistic program `Dist` in this DSL resembles a Sum--Product network.

Internal nodes are `Mixture` `Product` expressions.

Leaf nodes are either a `Primitive` distribution (whose name is `symbol` and has
a bundle of parameters `theta`) or a `Transform` of a `Primitive`.

The higher-order constructor `Condition` takes in an arbitrary `Dist` and a
probabilistic `event` specifying boolean predicates on the `symbol`s of the
`Primitive` distributions in the network and returns a new `Dist` representing
the conditional distribution given the event.

### Finding the probability of an event

Given a probabilistic program `dist`, the key query is finding the log
probability of a given `event`:

    (logprob dist event)

The conditional probability of an event is obtained by querying a conditioned
network, for example

    (logprob
            (condition
                (mixture
                    ((normal 'X 0 1) (gamma 'X 1 1))
                    (.7 .3))
                (< exp('X) 2)               ; Conditioned Event
            (or (< sqrt('X))                ; Queried Event
                (> (- 'X**2 'X/2) 1))))

If the cumulative probabilities of the `Primitive` distributions (on either
finite, countable, or uncountable domains) are known then exact inference in the
network is possible using symbolic analysis with fixed runtime.

### Finding the mutual information between events

Given a probabilistic program `dist`, the mutual information of
`eventA` and `eventB` given `eventC` corresponds to the following expression:

    (let
        ([distc (condition dist eventC)]
         [lpA1 (logprob distc eventA)]
         [lpB1 (logprob distc eventB)]
         [lpA0 (- 1 lpA1)]
         [lpB0 (- 1 lpB1)]
         [lp00 (logprob dist (and ((not eventA) (not eventB))))]
         [lp01 (logprob dist (and ((not eventA) eventB)))]
         [lp10 (logprob dist (and (eventA (not eventB))))]
         [lp11 (logprob dist (and (eventA eventB)))]
         [m00 (* (- lp00 (+ lpA0 lpB0 ) (exp lp00)))]
         [m01 (* (- lp01 (+ lpA0 lpB1 ) (exp lp01)))]
         [m10 (* (- lp10 (+ lpA1 lpB0 ) (exp lp10)))]
         [m11 (* (- lp11 (+ lpA1 lpB1 ) (exp lp11)))])
        ; Compute average
        (+ m00 m01 m10 m11))
