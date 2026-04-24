Examples
========

All examples are in the ``Examples/`` directory.

Pre-processing
--------------

.. code-block:: bash

   python Examples/preprocess/example_stl_basic.py

Uses the bundled test geometry ``Examples/preprocess/data/grid_of_cubes.stl``
(390×390×30 m domain).

Neural Network Inference
------------------------

.. code-block:: bash

   python Examples/nn/example_inference.py

Requires model weights in ``170625_weights/``.

Post-processing
---------------

.. code-block:: bash

   python Examples/postprocess/example_overlap.py
