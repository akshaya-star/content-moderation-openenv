import sys
sys.path.insert(0, 'src')

# Check routes explicitly
from server.app import app
routes = sorted([r.path for r in app.routes])
print('=== ROUTES ===')
for r in routes:
    print(' ', r)

# Check for the hard episode messages count issue
from content_moderation_env.server.utils import load_json_samples, pick_episode
hard = load_json_samples('hard')
print()
print(f'Hard samples: {len(hard)} top-level episodes')
ep = pick_episode(hard, 42, None)
msgs = ep.get('messages', [])
print(f'Picked episode id={ep.get("episode_id")}, messages={len(msgs)}')
print()
print('=== Hard episode message ids ===')
for m in msgs:
    print(f'  {m["id"]} -> {m["expected_decision"]}')

# Check the observation total_messages from reset
from content_moderation_env.server.environment import ContentModerationEnvironment
env = ContentModerationEnvironment()
obs = env.reset(seed=42, task='hard')
print()
print(f'=== reset(hard, seed=42) ===')
print(f'  total_messages={obs.total_messages}')
print(f'  current_message={obs.current_message}')
print(f'  done={obs.done}')
