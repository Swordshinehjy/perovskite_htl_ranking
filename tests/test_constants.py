import pytest
from htl_package.constants import (
    EXTRA_COLS,
    EXTRA_DIM,
    GLOBAL_COLS,
    GLOBAL_DIM,
    TASK_NAMES,
    NUM_TASKS,
)


class TestConstants:
    def test_extra_cols_is_list(self):
        assert isinstance(EXTRA_COLS, list)

    def test_extra_dim_matches_cols(self):
        assert EXTRA_DIM == len(EXTRA_COLS)

    def test_extra_cols_have_placeholder(self):
        for col in EXTRA_COLS:
            assert "{s}" in col

    def test_global_cols_is_list(self):
        assert isinstance(GLOBAL_COLS, list)

    def test_global_dim_matches_cols(self):
        assert GLOBAL_DIM == len(GLOBAL_COLS)

    def test_global_cols_contains_mo_ito(self):
        assert "MO_ITO" in GLOBAL_COLS

    def test_task_names_is_list(self):
        assert isinstance(TASK_NAMES, list)

    def test_num_tasks_matches_names(self):
        assert NUM_TASKS == len(TASK_NAMES)

    def test_task_names_contains_pce(self):
        assert "PCE" in TASK_NAMES

    def test_extra_dim_value(self):
        assert EXTRA_DIM == 13

    def test_global_dim_value(self):
        assert GLOBAL_DIM == 1

    def test_num_tasks_value(self):
        assert NUM_TASKS == 1
