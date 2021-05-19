# from safe_rl import ppo_lagrangian
# import gym, safety_gym
#
# ppo_lagrangian(
# 	env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
# 	ac_kwargs = dict(hidden_sizes=(64,64))
# 	)
#
#

print("select  queryid,\n" +
				"       city_id,\n" +
				"       app_version,\n" +
				"       lat,\n" +
				"       lng,\n" +
				"       os,\n" +
				"       network,\n" +
				"       hour,\n" +
				"       weekday,\n" +
				"       slotid,\n" +
				"       app_id, \n" +
				"       os_version, \n" +
				"       device_model, \n" +
				"       extend['deviceid'], \n" +
				"       extend['launch_scene'], \n" +
				"       extend['is_dp_install'] \n" +
				"  from mart_dprec.adx_ctr_context\n" +
				" where hp_cal_dt='<bizdate_wide>'\n")