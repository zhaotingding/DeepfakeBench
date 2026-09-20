"""CPU-only fault injection; never calls Codex or launches training."""
import contextlib
import errno
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import watch_prd_early_stop as watch


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        log = root / 'training.log'
        log.write_text('interrupted training')
        self.state = root / 'trigger.json'
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for key, value in [('STATE', root), ('LOG', log), ('ROOT', root)]:
            self.stack.enter_context(patch.object(watch, key, value))
        self.stack.enter_context(patch.object(watch, 'training_alive', return_value=False))
        self.active = self.stack.enter_context(patch.object(watch, 'project_training', return_value=[]))
        self.stack.enter_context(patch.object(watch, 'write_experiment_summary', return_value='summary.md'))
        self.call = self.stack.enter_context(patch.object(watch, 'dispatch', return_value=0))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def retry_now(self):
        job = json.loads(self.state.read_text())
        job['retry_after'] = 0
        watch.save_json(self.state, job)

    def test_abnormal_is_not_normal(self):
        self.assertIn('异常退出', watch.handoff_reason('', [], False))
        self.assertIsNone(watch.handoff_reason('', [], True))
        self.assertIn('正常完成', watch.handoff_reason('Stop Training on best Testing metric', [], False))

    def test_declines(self):
        self.assertEqual(watch.trigger_epoch(list(enumerate([.9, .8, .7, .6]))), 3)
        self.assertIsNone(watch.trigger_epoch(list(enumerate([.9, .8, .8, .7]))))

    def test_observe_never_dispatches(self):
        watch.tick(False, -1)
        self.call.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_failure_retry_success_not_repeated(self):
        self.call.side_effect = [1, 0]
        self.active.side_effect = [[], [], [], [456], []]
        watch.tick(True, -1)
        watch.tick(True, -1)
        self.assertEqual(self.call.call_count, 1)
        self.retry_now()
        watch.tick(True, -1)
        watch.tick(True, -1)
        self.assertEqual(self.call.call_count, 2)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_returned')

    def test_retry_exhaustion(self):
        self.call.side_effect = OSError('injected spawn failure')
        for _ in range(4):
            watch.tick(True, -1)
            self.retry_now()
        self.assertEqual(self.call.call_count, 3)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'exhausted')

    def test_recover_interrupted_dispatch(self):
        watch.save_json(self.state, {'phase': 'dispatching', 'attempts': 1})
        watch.tick(True, -1)
        self.assertEqual(self.call.call_count, 1)
        self.assertEqual(json.loads(self.state.read_text())['attempts'], 2)

    def test_existing_training_blocks_retry(self):
        watch.save_json(self.state, {'phase': 'codex_failed', 'attempts': 1})
        self.active.return_value = [123]
        watch.tick(True, -1)
        self.call.assert_not_called()
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'blocked_active_training')

    def test_network_filesystem_lock_contention(self):
        with patch.object(watch.sys, 'argv', ['watch', '--once', '--dispatch']), \
                patch.object(watch.fcntl, 'flock', side_effect=PermissionError(errno.EACCES, 'locked')):
            watch.main()
        self.call.assert_not_called()

    def test_live_training_without_trigger(self):
        with patch.object(watch, 'training_alive', return_value=True):
            watch.tick(True, -1)
        self.call.assert_not_called()

    def test_declining_training_must_finish_before_handoff(self):
        rows = list(enumerate([.9, .8, .7, .6]))
        self.assertIsNone(watch.handoff_reason('', rows, True))
        self.assertIn('自动早停', watch.handoff_reason(
            '[EarlyStop] retaining saved best checkpoints\nStop Training on best Testing metric', rows, False))

    def test_abnormal_report_only_is_not_recovery(self):
        watch.tick(True, -1)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_failed')

    def test_normal_handoff_without_next_training_is_failure(self):
        watch.LOG.write_text('Stop Training on best Testing metric')
        watch.tick(True, -1)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_failed')

    def test_normal_handoff_with_next_training(self):
        watch.LOG.write_text('Stop Training on best Testing metric')
        self.active.side_effect = [[], [456]]
        watch.tick(True, -1)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_returned')

    def test_final_queue_complete_without_next_training(self):
        watch.LOG.write_text('Stop Training on best Testing metric')
        def finish(*args):
            watch.save_json(watch.STATE / 'queue_complete.json',
                            {'experiment': 'F2', 'log': str(watch.LOG), 'status': 'complete'})
            return 0
        self.call.side_effect = finish
        watch.tick(True, -1)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_returned')
        watch.tick(True, -1)
        self.assertEqual(self.call.call_count, 1)

    def test_completion_marker_cannot_hide_abnormal_exit(self):
        watch.save_json(watch.STATE / 'queue_complete.json',
                        {'experiment': 'F2', 'log': str(watch.LOG), 'status': 'complete'})
        watch.tick(True, -1)
        self.assertEqual(json.loads(self.state.read_text())['phase'], 'codex_failed')


if __name__ == '__main__':
    unittest.main()
