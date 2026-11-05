import ast
import datetime
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def functions(relative, names, **context):
    source=ast.parse((ROOT/relative).read_text(encoding="utf-8"))
    nodes=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)]
    nodes.extend(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names)
    namespace={"pd":pd,"datetime":datetime,"sys":sys,**context}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),relative,"exec"),namespace)
    return namespace

class MigrationTests(unittest.TestCase):
    def test_full_industry_mapping_uses_constituents(self):
        sdk=Mock()
        sdk.get_industry_constituents.return_value=pd.DataFrame([{"stock_symbol":"000001.SZ","industry_code":"801780"}])
        ns=functions("scripts/data/loader.py",{"load_stock_industry"},_safe_date_col=lambda x:x,EXPECTED_COLUMNS={"stock_industry":set()})
        with patch.dict(sys.modules,{"panda_data":sdk}):
            result=ns["load_stock_industry"]()
        sdk.get_stock_industry.assert_not_called()
        sdk.get_industry_constituents.assert_called_once()
        self.assertEqual(len(result),1)

    def test_financial_request_has_quarters_and_asof(self):
        sdk=Mock()
        sdk.get_fina_reports.return_value=pd.DataFrame([{"symbol":"000001.SZ","date":"20260430"}])
        cache=Mock();cache.read_cache.return_value=None
        ns=functions("scripts/data/loader.py",{"load_fina_reports"},cache=cache,_safe_date_col=lambda x:x,EXPECTED_COLUMNS={"fina_reports":set()})
        with patch.dict(sys.modules,{"panda_data":sdk}):
            result=ns["load_fina_reports"](["000001.SZ"],"20250101","20260630")
        params=sdk.get_fina_reports.call_args.kwargs
        self.assertEqual((params["start_quarter"],params["end_quarter"]),("2025q1","2026q2"))
        self.assertEqual(params["date"],"20260630")
        self.assertEqual(len(result),1)

if __name__ == "__main__":
    unittest.main()
