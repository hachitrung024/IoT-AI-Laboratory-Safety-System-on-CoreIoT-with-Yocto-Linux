// SPDX-License-Identifier: GPL-2.0-only

/*
 * dht20.c - Linux hwmon driver for DHT20 Temperature and Humidity sensor
 */

#include <linux/crc8.h>
#include <linux/delay.h>
#include <linux/hwmon.h>
#include <linux/i2c.h>
#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/version.h>

#define DHT20_MEAS_SIZE			7
#define DHT20_CRC8_POLY			0x31

/*
 * Poll intervals (in milliseconds)
 */
#define DHT20_DEFAULT_MIN_POLL_INTERVAL	2000
#define DHT20_MIN_POLL_INTERVAL			2000

/*
 * I2C command delays (in microseconds)
 */
#define DHT20_MEAS_DELAY	80000
#define DHT20_DELAY_EXTRA	100000

/*
 * Command bytes
 */
#define DHT20_CMD_MEAS		0xAC

/*
 * Flags in sensor status byte
 */
#define DHT20_BUSY		BIT(7)

DECLARE_CRC8_TABLE(dht20_crc8_table);

struct dht20_data {
	struct i2c_client *client;
	/* Protect shared sensor state and poll timing. */
	struct mutex lock;
	ktime_t min_poll_interval;
	ktime_t previous_poll_time;
	int temperature;
	int humidity;
};

static bool dht20_polltime_expired(struct dht20_data *data)
{
	ktime_t current_time = ktime_get_boottime();
	ktime_t difference = ktime_sub(current_time, data->previous_poll_time);

	return ktime_after(difference, data->min_poll_interval);
}

/*
 * crc calculated on the whole frame (including crc byte) should yield zero
 * in case of correctly received bytes.
 */
static int dht20_crc8_check(u8 *raw_data, int count)
{
	return crc8(dht20_crc8_table, raw_data, count, CRC8_INIT_VALUE);
}

static int dht20_read_values(struct dht20_data *data)
{
	const u8 cmd_meas[] = { DHT20_CMD_MEAS, 0x33, 0x00 };
	u32 temp, hum;
	int res;
	u8 raw_data[DHT20_MEAS_SIZE];
	struct i2c_client *client = data->client;

	mutex_lock(&data->lock);
	if (!dht20_polltime_expired(data)) {
		mutex_unlock(&data->lock);
		return 0;
	}

	res = i2c_master_send(client, cmd_meas, sizeof(cmd_meas));
	if (res < 0) {
		mutex_unlock(&data->lock);
		return res;
	}

	usleep_range(DHT20_MEAS_DELAY, DHT20_MEAS_DELAY + DHT20_DELAY_EXTRA);

	res = i2c_master_recv(client, raw_data, DHT20_MEAS_SIZE);
	if (res != DHT20_MEAS_SIZE) {
		mutex_unlock(&data->lock);
		if (res >= 0)
			return -ENODATA;
		return res;
	}

	if (raw_data[0] & DHT20_BUSY) {
		mutex_unlock(&data->lock);
		return -EBUSY;
	}

	if (dht20_crc8_check(raw_data, DHT20_MEAS_SIZE)) {
		mutex_unlock(&data->lock);
		return -EIO;
	}

	hum = ((u32)raw_data[1] << 12u) |
	      ((u32)raw_data[2] << 4u) |
	      ((raw_data[3] & 0xF0u) >> 4u);

	temp = ((u32)(raw_data[3] & 0x0Fu) << 16u) |
	       ((u32)raw_data[4] << 8u) |
	       raw_data[5];

	temp = ((temp * 625) >> 15u) * 10;
	hum = ((hum * 625) >> 16u) * 10;

	data->temperature = (int)temp - 50000;
	data->humidity = hum;
	data->previous_poll_time = ktime_get_boottime();

	mutex_unlock(&data->lock);
	return 0;
}

static ssize_t dht20_interval_write(struct dht20_data *data, long val)
{
	data->min_poll_interval = ms_to_ktime(clamp_val(val, DHT20_MIN_POLL_INTERVAL,
						LONG_MAX));
	return 0;
}

static ssize_t dht20_interval_read(struct dht20_data *data, long *val)
{
	*val = ktime_to_ms(data->min_poll_interval);
	return 0;
}

static int dht20_temperature1_read(struct dht20_data *data, long *val)
{
	int res;

	res = dht20_read_values(data);
	if (res < 0)
		return res;

	*val = data->temperature;
	return 0;
}

static int dht20_humidity1_read(struct dht20_data *data, long *val)
{
	int res;

	res = dht20_read_values(data);
	if (res < 0)
		return res;

	*val = data->humidity;
	return 0;
}

static umode_t dht20_hwmon_visible(const void *data,
				   enum hwmon_sensor_types type,
				   u32 attr, int channel)
{
	switch (type) {
	case hwmon_temp:
	case hwmon_humidity:
		return 0444;
	case hwmon_chip:
		return 0644;
	default:
		return 0;
	}
}

static int dht20_hwmon_read(struct device *dev, enum hwmon_sensor_types type,
			    u32 attr, int channel, long *val)
{
	struct dht20_data *data = dev_get_drvdata(dev);

	switch (type) {
	case hwmon_temp:
		return dht20_temperature1_read(data, val);
	case hwmon_humidity:
		return dht20_humidity1_read(data, val);
	case hwmon_chip:
		return dht20_interval_read(data, val);
	default:
		return -EOPNOTSUPP;
	}
}

static int dht20_hwmon_write(struct device *dev, enum hwmon_sensor_types type,
			     u32 attr, int channel, long val)
{
	struct dht20_data *data = dev_get_drvdata(dev);

	switch (type) {
	case hwmon_chip:
		return dht20_interval_write(data, val);
	default:
		return -EOPNOTSUPP;
	}
}

static const struct hwmon_channel_info * const dht20_info[] = {
	HWMON_CHANNEL_INFO(chip, HWMON_C_UPDATE_INTERVAL),
	HWMON_CHANNEL_INFO(temp, HWMON_T_INPUT),
	HWMON_CHANNEL_INFO(humidity, HWMON_H_INPUT),
	NULL,
};

static const struct hwmon_ops dht20_hwmon_ops = {
	.is_visible = dht20_hwmon_visible,
	.read = dht20_hwmon_read,
	.write = dht20_hwmon_write,
};

static const struct hwmon_chip_info dht20_chip_info = {
	.ops = &dht20_hwmon_ops,
	.info = dht20_info,
};

static const struct i2c_device_id dht20_id[] = {
	{ "dht20", 0 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, dht20_id);

static const struct of_device_id dht20_of_match[] = {
	{ .compatible = "asair,dht20" },
	{ }
};
MODULE_DEVICE_TABLE(of, dht20_of_match);

static int dht20_probe(struct i2c_client *client)
{
	struct device *device = &client->dev;
	struct device *hwmon_dev;
	struct dht20_data *data;
	int res;

	if (!i2c_check_functionality(client->adapter, I2C_FUNC_I2C))
		return -ENOENT;

	data = devm_kzalloc(device, sizeof(*data), GFP_KERNEL);
	if (!data)
		return -ENOMEM;

	data->client = client;
	data->min_poll_interval = ms_to_ktime(DHT20_DEFAULT_MIN_POLL_INTERVAL);
	/* Make the first read available immediately after probe. */
	data->previous_poll_time = ktime_sub(ktime_get_boottime(),
					    data->min_poll_interval);

	mutex_init(&data->lock);
	crc8_populate_msb(dht20_crc8_table, DHT20_CRC8_POLY);

	res = dht20_read_values(data);
	if (res < 0)
		return res;

	hwmon_dev = devm_hwmon_device_register_with_info(device,
						 client->name,
						 data,
						 &dht20_chip_info,
						 NULL);

	return PTR_ERR_OR_ZERO(hwmon_dev);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 1, 0)
static void dht20_remove(struct i2c_client *client)
{
}
#else
static int dht20_remove(struct i2c_client *client)
{
	return 0;
}
#endif

static struct i2c_driver dht20_driver = {
	.driver = {
		.name = "dht20",
		.of_match_table = dht20_of_match,
	},
	.probe = dht20_probe,
	.remove = dht20_remove,
	.id_table = dht20_id,
};

module_i2c_driver(dht20_driver);

MODULE_AUTHOR("Trung Ha <hachitrung024@gmail.com>");
MODULE_DESCRIPTION("DHT20 Temperature and Humidity sensor driver");
MODULE_VERSION("1.0");
MODULE_LICENSE("GPL v2");
